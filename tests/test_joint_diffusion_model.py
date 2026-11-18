"""Joint visual/Llama LoRA training and portable adapter checkpoint regressions."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from src.model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
from src.model_wrapper.diffusion_action_head import DiffusionActionHeadConfig


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, pixels):
        return self.projection(pixels.mean(dim=(-2, -1))).unsqueeze(1).expand(-1, 2, -1)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = TinyVision()
        self.projector = nn.Linear(8, 16)
        self.language_model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
            attention_dropout=0.0, pad_token_id=0, bos_token_id=1,
        ))
        self.pad_token_id = 0

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()


def make_model(backbone=None):
    return AeroVLADiffusionModel(
        backbone or TinyBackbone(),
        DiffusionActionHeadConfig(condition_dim=16, hidden_dim=16, num_layers=1,
                                  num_heads=4, num_train_timesteps=8, num_inference_steps=2),
        base_model_path="/test/base", adapter_path="/test/original_adapter",
    )


def observations():
    return {"input_ids": torch.tensor([[1, 3, 5], [1, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
            "pixel_values": torch.randn(2, 3, 4, 4)}


def group(name):
    if name.startswith("diffusion_head."):
        return "head"
    if "projector" in name:
        return "projector"
    if "vision_backbone" in name:
        return "vision"
    return "language"


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_joint_step_updates_all_four_groups_and_keeps_base_frozen(gradient_checkpointing):
    torch.manual_seed(7)
    model = make_model().enable_joint_lora(rank=2, alpha=4, gradient_checkpointing=gradient_checkpointing).train()
    trainable = {name: value for name, value in model.named_parameters() if value.requires_grad}
    assert {group(name) for name in trainable} == {"head", "projector", "vision", "language"}
    assert all(value.dtype == torch.float32 for value in trainable.values())
    assert all(name.startswith("diffusion_head.") or "lora_" in name or "modules_to_save.default" in name for name in trainable)
    frozen_before = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
    trained_before = {name: value.clone() for name, value in trainable.items()}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.01, weight_decay=0)
    batch = observations()
    condition, _ = model.encode_condition(**batch)
    assert condition.requires_grad
    # LoRA B starts at zero: a second update also checks that A receives signal.
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, actions=torch.zeros(2, 1, 4),
                     noise=torch.ones(2, 1, 4), timesteps=torch.tensor([2, 5]))
        assert torch.isfinite(loss)
        loss.backward()
        assert {group(name) for name, value in trainable.items()
                if value.grad is not None and value.grad.abs().sum() > 0} == {"head", "projector", "vision", "language"}
        optimizer.step()
    assert {group(name) for name, value in trainable.items()
            if not torch.equal(value, trained_before[name])} == {"head", "projector", "vision", "language"}
    for name, value in model.named_parameters():
        if name in frozen_before:
            assert value.grad is None
            torch.testing.assert_close(value, frozen_before[name], rtol=0, atol=0)
    model.eval()
    assert not model.encode_condition(**batch)[0].requires_grad
    model.train()
    assert model.backbone.training
    assert set(trainable) == {name for name, value in model.named_parameters() if value.requires_grad}


def test_joint_checkpoint_independent_reload_preserves_predictions_and_adapters(tmp_path):
    torch.manual_seed(19)
    pristine = TinyBackbone()
    model = make_model(copy.deepcopy(pristine)).enable_joint_lora(rank=2, alpha=4).train()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.02)
    batch = observations()
    model(**batch, actions=torch.zeros(2, 1, 4)).backward()
    optimizer.step()
    model.eval().save_pretrained(tmp_path, training_metadata={"step": 1})
    config = model.read_checkpoint_config(tmp_path)
    assert config["format_version"] == 2
    state = torch.load(tmp_path / model.JOINT_WEIGHTS_NAME, weights_only=True)
    assert state and all("lora_" in name or "projector" in name for name in state)
    assert not any("base_layer" in name or "original_module" in name for name in state)
    with patch.object(AeroVLADiffusionModel, "from_pretrained_backbone",
                      side_effect=lambda *args, **kwargs: make_model(copy.deepcopy(pristine))):
        restored = AeroVLADiffusionModel.from_pretrained(tmp_path, torch_dtype=torch.float32, device="cpu")
    assert not restored.training
    assert restored.joint_lora_config == model.joint_lora_config
    expected = model.predict_action(**batch, generator=torch.Generator().manual_seed(35))
    actual = restored.predict_action(**batch, generator=torch.Generator().manual_seed(35))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    before = {name: value.clone() for name, value in restored.named_parameters()}
    restored.enable_joint_lora(rank=2, alpha=4, gradient_checkpointing=False).train()
    for name, value in restored.named_parameters():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not restored.backbone.language_model.get_decoder().gradient_checkpointing
    with pytest.raises(ValueError, match="refusing to reset"):
        restored.enable_joint_lora(rank=3, alpha=4)


def test_joint_checkpoint_missing_adapter_tensor_rejected(tmp_path):
    pristine = TinyBackbone()
    source = make_model(copy.deepcopy(pristine)).enable_joint_lora(rank=2, alpha=4)
    source.save_pretrained(tmp_path)
    state = torch.load(tmp_path / source.JOINT_WEIGHTS_NAME, weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, tmp_path / source.JOINT_WEIGHTS_NAME)
    target = make_model(copy.deepcopy(pristine))
    with pytest.raises(ValueError, match="missing, extra"):
        target._load_joint_checkpoint(tmp_path, source.read_checkpoint_config(tmp_path))
    config_path = tmp_path / source.CONFIG_NAME
    config = json.loads(config_path.read_text())
    config["joint_lora_config"]["rank"] = 0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="joint_lora_config"):
        source.read_checkpoint_config(tmp_path)


def test_joint_bfloat16_base_has_float32_trainables_and_projector_inputs():
    model = make_model(TinyBackbone().to(dtype=torch.bfloat16)).enable_joint_lora(rank=2, alpha=4).train()
    batch = observations()
    loss = model(**batch, actions=torch.zeros(2, 1, 4))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.dtype == torch.float32 for value in model.trainable_parameters())
    assert model.backbone.get_input_embeddings().weight.dtype == torch.bfloat16
