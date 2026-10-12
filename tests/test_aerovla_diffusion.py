"""Exercise a real tiny Llama decoder without downloading or loading OpenVLA-7B."""

import copy
import json
from unittest.mock import patch

import pytest
import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from src.model_wrapper.aerovla_diffusion import AeroVLADiffusionModel, DEFAULT_ACTION_SCHEMA
from src.model_wrapper.diffusion_action_head import DiffusionActionHeadConfig


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.8)

    def forward(self, pixels):
        features = pixels.mean(dim=(-2, -1))
        return self.dropout(self.projection(features)).unsqueeze(1).expand(-1, 2, -1)


class TinyPrismatic(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=64, attention_dropout=0.5,
            pad_token_id=0, bos_token_id=1,
        )
        self.vision_backbone = TinyVision()
        self.projector = nn.Linear(8, 16)
        self.language_model = LlamaForCausalLM(self.config)
        self.pad_token_id = 0

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def resize_token_embeddings(self, size):
        return self.language_model.resize_token_embeddings(size)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.language_model.prepare_inputs_for_generation(*args, **kwargs)


def make_model(backbone=None, **overrides):
    config = dict(condition_dim=16, hidden_dim=16, num_layers=1, num_heads=4,
                  num_train_timesteps=8, num_inference_steps=2, action_horizon=2)
    config.update(overrides)
    return AeroVLADiffusionModel(backbone or TinyPrismatic(), DiffusionActionHeadConfig(**config))


def inputs():
    return {
        "input_ids": torch.tensor([[1, 5, 6, 7], [1, 8, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        "pixel_values": torch.randn(2, 3, 4, 4),
    }


def test_frozen_real_llama_and_head_only_optimizer_step():
    torch.manual_seed(20)
    model = make_model()
    model.train()
    assert model.training and model.diffusion_head.training
    assert all(not module.training for module in model.backbone.modules())
    assert not hasattr(model.backbone.language_model, "lm_head")
    assert not any("lm_head" in name for name, _ in model.named_parameters())
    assert all(not value.requires_grad for value in model.backbone.parameters())
    assert all(name.startswith("diffusion_head.") for name, value in model.named_parameters() if value.requires_grad)
    before_backbone = {name: value.clone() for name, value in model.backbone.state_dict().items()}
    before_head = {name: value.clone() for name, value in model.diffusion_head.state_dict().items()}
    batch = inputs()
    first, _ = model.encode_condition(**batch)
    second, _ = model.encode_condition(**batch)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not first.requires_grad
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    loss = model(**batch, actions=torch.zeros(2, 2, 4), timesteps=torch.tensor([1, 6]))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(value.grad is None for value in model.backbone.parameters())
    assert any(value.grad is not None and value.grad.abs().sum() > 0 for value in model.diffusion_head.parameters())
    optimizer.step()
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, before_backbone[name], rtol=0, atol=0)
    assert any(not torch.equal(value, before_head[name]) for name, value in model.diffusion_head.state_dict().items())


def test_multimodal_insertion_mask_and_padded_tokens_are_ignored():
    model = make_model().eval()
    batch = inputs()
    decoder = model.backbone.language_model.get_decoder()
    with patch.object(decoder, "forward", wraps=decoder.forward) as forward:
        condition, mask = model.encode_condition(**batch)
    expected_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    assert condition.shape == (2, 6, 16)
    torch.testing.assert_close(mask, expected_mask)
    arguments = forward.call_args.kwargs
    embedding = model.backbone.get_input_embeddings()(batch["input_ids"])
    projected = model.backbone.projector(model.backbone.vision_backbone(batch["pixel_values"]))
    expected_embedding = torch.cat((embedding[:, :1], projected, embedding[:, 1:]), dim=1)
    torch.testing.assert_close(arguments["inputs_embeds"], expected_embedding)
    assert arguments["use_cache"] is False
    changed = copy.deepcopy(batch)
    changed["input_ids"][1, 2:] = torch.tensor([22, 23])
    changed_condition, _ = model.encode_condition(**changed)
    torch.testing.assert_close(condition[mask], changed_condition[mask], rtol=1e-5, atol=1e-6)
    a = model.predict_action(**batch, generator=torch.Generator().manual_seed(42))
    b = model.predict_action(**changed, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_mask_inferred_from_pad_tokens_and_left_padding_rejected():
    model = make_model()
    batch = inputs()
    _, mask = model.encode_condition(batch["input_ids"], batch["pixel_values"])
    assert mask[1].tolist() == [True, True, True, True, False, False]
    batch["attention_mask"][1] = torch.tensor([0, 0, 1, 1])
    with pytest.raises(ValueError, match="right-padded"):
        model.encode_condition(**batch)


def test_head_only_checkpoint_round_trip_and_metadata(tmp_path):
    torch.manual_seed(7)
    source = make_model().eval()
    source.base_model_path = "/example/base"
    source.adapter_path = "/example/adapter"
    target = make_model(copy.deepcopy(source.backbone)).eval()
    source.save_pretrained(tmp_path, training_metadata={"global_step": 3})
    config = json.loads((tmp_path / source.CONFIG_NAME).read_text())
    assert config["action_schema"] == DEFAULT_ACTION_SCHEMA
    assert config["training_metadata"] == {"global_step": 3}
    state = torch.load(tmp_path / source.WEIGHTS_NAME, weights_only=True)
    assert set(state) == set(source.diffusion_head.state_dict())
    target.load_head_checkpoint(tmp_path)
    batch = inputs()
    prediction = source.predict_action(**batch, generator=torch.Generator().manual_seed(8))
    restored = target.predict_action(**batch, generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(prediction, restored, rtol=0, atol=0)
    with patch.object(AeroVLADiffusionModel, "from_pretrained_backbone", return_value=target) as loader:
        loaded = AeroVLADiffusionModel.from_pretrained(tmp_path, torch_dtype=torch.float32, device="cpu")
    assert loaded is target and not loaded.training
    assert loader.call_args.args == ("/example/base",)
    assert loader.call_args.kwargs["adapter_path"] == "/example/adapter"
    assert loader.call_args.kwargs["tokenizer_length"] == 32


def test_checkpoint_rejects_schema_config_and_incomplete_weights(tmp_path):
    model = make_model()
    model.save_pretrained(tmp_path)
    incompatible = make_model(action_horizon=1)
    with pytest.raises(ValueError, match="configuration"):
        incompatible.load_head_checkpoint(tmp_path)
    state = torch.load(tmp_path / model.WEIGHTS_NAME, weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, tmp_path / model.WEIGHTS_NAME)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_head_checkpoint(tmp_path)
    config_path = tmp_path / model.CONFIG_NAME
    config = json.loads(config_path.read_text())
    config["action_schema"]["action_order"] = ["down", "fwd", "yaw", "land"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="action_schema"):
        model.load_head_checkpoint(tmp_path)


def test_actual_peft_merge_preserves_adapter_projector_and_lora(tmp_path):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(13)
    pristine = TinyPrismatic()
    backbone = copy.deepcopy(pristine)
    peft_model = get_peft_model(backbone, LoraConfig(
        r=2, lora_alpha=2, target_modules=["q_proj"],
        modules_to_save=["projector"], task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        peft_model.base_model.model.projector.modules_to_save["default"].weight.fill_(0.75)
        peft_model.base_model.model.projector.modules_to_save["default"].bias.fill_(0.25)
        query = peft_model.base_model.model.language_model.model.layers[0].self_attn.q_proj
        query.lora_A["default"].weight.fill_(0.2)
        query.lora_B["default"].weight.fill_(0.3)
        expected_query = query.base_layer.weight.clone() + query.get_delta_weight("default")
    peft_model.save_pretrained(tmp_path)
    config = DiffusionActionHeadConfig(condition_dim=16, hidden_dim=16, num_layers=1, num_heads=4)
    with patch("transformers.AutoModelForVision2Seq.from_pretrained", return_value=pristine):
        result = AeroVLADiffusionModel.from_pretrained_backbone(
            "tiny-base", adapter_path=tmp_path, head_config=config,
            torch_dtype=torch.float32, device="cpu", tokenizer_length=32,
        )
    torch.testing.assert_close(result.backbone.projector.weight, torch.full((16, 8), 0.75))
    torch.testing.assert_close(result.backbone.projector.bias, torch.full((16,), 0.25))
    torch.testing.assert_close(result.backbone.language_model.model.layers[0].self_attn.q_proj.weight, expected_query)
    assert all(not parameter.requires_grad for parameter in result.backbone.parameters())
    assert not any("lora_" in name or "modules_to_save" in name for name, _ in result.named_parameters())


def test_condition_dimension_validation():
    with pytest.raises(ValueError, match="condition_dim"):
        make_model(condition_dim=32)
    with pytest.raises(ValueError, match="four action channels"):
        make_model(action_dim=3)
