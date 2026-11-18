"""Offline validation checks with hand-computed errors and a real tiny DDPM head."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.eval_diffusion_head import build_parser, evaluate_diffusion, validate_checkpoint_split
from src.model_wrapper.diffusion_action_head import DiffusionActionHead, DiffusionActionHeadConfig


class FixedHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(num_inference_steps=2, num_train_timesteps=8)
        self.loss_calls = 0
        self.sample_calls = 0

    def compute_loss(self, condition, actions, *, action_mask, noise, timesteps, condition_mask):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert noise.shape == actions.shape
        assert timesteps.shape == (len(actions),)
        self.loss_calls += 1
        # Known per-coordinate errors expose incorrect averaging of a short tail.
        return actions.masked_select(action_mask).square().mean()

    def sample(self, condition, *, generator, **kwargs):
        assert not torch.is_grad_enabled()
        assert isinstance(generator, torch.Generator)
        self.sample_calls += 1
        return condition + self.weight * 0


class FixedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(4, 4).requires_grad_(False)
        self.backbone.eval()
        self.diffusion_head = FixedHead()
        self.encode_calls = 0

    def encode_condition(self, input_ids, pixel_values, attention_mask):
        assert not torch.is_grad_enabled()
        assert not self.training and not self.backbone.training
        self.encode_calls += 1
        return pixel_values, attention_mask.bool()


def records():
    targets = [[-1, 0, 0, 1], [1, 1, 1, -1], [0, -1, -1, 1]]
    predictions = [[-1, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, -1]]
    return [{"input_ids": torch.tensor([1]), "attention_mask": torch.tensor([1]),
             "pixel_values": torch.tensor([prediction], dtype=torch.float32),
             "actions": torch.tensor([target], dtype=torch.float32)}
            for target, prediction in zip(targets, predictions)]


def test_hand_computed_action_land_metrics_and_short_tail():
    model = FixedPolicy()
    loader = DataLoader(records(), batch_size=2)
    result = evaluate_diffusion(model, loader, device="cpu")
    assert result["num_samples"] == 3
    assert result["num_valid_coordinates"] == 12
    assert result["noise_mse"] == pytest.approx(0.75)
    assert result["normalized_continuous_mse"] == pytest.approx(12 / 9)
    assert result["physical"]["fwd"]["mae"] == pytest.approx(5 / 3)
    assert result["physical"]["fwd"]["rmse"] == pytest.approx(math.sqrt(12.5 / 3))
    assert result["physical"]["down"]["mae"] == pytest.approx(5)
    assert result["physical"]["down"]["rmse"] == pytest.approx(math.sqrt(125 / 3))
    assert result["physical"]["yaw"]["mae"] == pytest.approx(1.1)
    assert result["physical"]["yaw"]["rmse"] == pytest.approx(math.sqrt(6.05 / 3))
    assert {key: result["land"][key] for key in ("tp", "tn", "fp", "fn")} == {
        "tp": 1, "tn": 0, "fp": 1, "fn": 1}
    assert result["land"]["precision"] == result["land"]["recall"] == result["land"]["f1"] == 0.5
    assert result["land"]["accuracy"] == pytest.approx(1 / 3)
    assert result["land"]["support_positive"] == 2
    assert result["land"]["support_negative"] == 1
    assert model.encode_calls == model.diffusion_head.loss_calls == model.diffusion_head.sample_calls == 2
    json.dumps(result, allow_nan=False)


def test_coordinate_masks_and_clipping_use_actual_coordinate_counts():
    data = records()
    data[0]["action_mask"] = torch.tensor([[0, 0, 0, 0]])
    data[0]["actions"].fill_(float("nan"))
    data[1]["action_mask"] = torch.tensor([[1, 0, 0, 0]])
    data[1]["pixel_values"][0, 0] = -9  # Saturates at physical fwd=0.
    data[2]["action_mask"] = torch.tensor([[0, 0, 0, 1]])
    result = evaluate_diffusion(FixedPolicy(), DataLoader(data, batch_size=2), device="cpu")
    assert result["num_samples"] == 3 and result["num_valid_samples"] == 2
    assert result["num_valid_coordinates"] == 2
    assert result["noise_mse"] == pytest.approx(1.0)
    assert result["normalized_continuous_mse"] == pytest.approx(4.0)
    assert result["physical"]["fwd"]["count"] == 1
    assert result["physical"]["fwd"]["mae"] == 5
    assert result["physical"]["down"]["mae"] is None
    assert result["land"]["fn"] == result["land"]["support_positive"] == 1
    assert result["land"]["support_negative"] == 0
    json.dumps(result, allow_nan=False)


def test_all_negative_land_support_prevents_accuracy_only_reporting():
    data = records()
    for sample in data:
        sample["actions"][0, 3] = -1
        sample["pixel_values"][0, 3] = -1
    result = evaluate_diffusion(FixedPolicy(), DataLoader(data, batch_size=2), device="cpu")
    land = result["land"]
    assert land["accuracy"] == 1
    assert land["precision"] == land["recall"] == land["f1"] == 0
    assert land["support_positive"] == 0 and land["support_negative"] == 3


class TinyDiffusionPolicy(FixedPolicy):
    def __init__(self):
        super().__init__()
        self.diffusion_head = DiffusionActionHead(DiffusionActionHeadConfig(
            condition_dim=4, hidden_dim=8, num_layers=1, num_heads=2,
            num_train_timesteps=8, num_inference_steps=2, dropout=0.8))

    def encode_condition(self, input_ids, pixel_values, attention_mask):
        features, mask = super().encode_condition(input_ids, pixel_values, attention_mask)
        return self.backbone(features), mask


def test_real_ddpm_reproducible_without_changing_rng_modes_weights_or_gradients():
    model = TinyDiffusionPolicy()
    # Mixed pre-existing modes must survive validation exactly.
    model.diffusion_head.output_norm.eval()
    modes = [module.training for module in model.modules()]
    state = copy.deepcopy(model.state_dict())
    loader = DataLoader(records(), batch_size=2, generator=torch.Generator().manual_seed(9))
    rng = torch.get_rng_state().clone()
    loader_rng = loader.generator.get_state().clone()
    first = evaluate_diffusion(model, loader, device="cpu", seed=42)
    second = evaluate_diffusion(model, loader, device="cpu", seed=42)
    assert first == second
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(loader.generator.get_state(), loader_rng)
    assert [module.training for module in model.modules()] == modes
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    noise_only = evaluate_diffusion(model, loader, device="cpu", seed=42, sample_actions=False)
    assert noise_only["noise_mse"] == first["noise_mse"]
    assert "physical" not in noise_only and "land" not in noise_only
    changed = evaluate_diffusion(model, loader, device="cpu", seed=43)
    assert changed["noise_mse"] != first["noise_mse"]


def test_evaluation_error_restores_modes_and_rng():
    model = FixedPolicy()
    original_modes = [module.training for module in model.modules()]
    data = records()
    data[0]["actions"].fill_(float("nan"))
    rng = torch.get_rng_state().clone()
    with pytest.raises(FloatingPointError, match="noise loss"):
        evaluate_diffusion(model, DataLoader(data, batch_size=2), device="cpu")
    assert torch.equal(torch.get_rng_state(), rng)
    assert [module.training for module in model.modules()] == original_modes
    with pytest.raises(ValueError, match="at least one valid"):
        evaluate_diffusion(model, [], device="cpu")


def test_checkpoint_provenance_rejects_mismatched_hashes_and_training_on_val(tmp_path):
    manifest = str(tmp_path / "data.json")
    split = {"split": "val", "data_sha256": "data", "split_sha256": "split", "trajectory_sha256": "traj"}
    saved = {**split, "split": "train", "data_json": manifest}
    config = {"training_metadata": {"train_split": saved, "train_json": manifest}}
    assert validate_checkpoint_split(config, split, manifest)["status"] == "verified_diffusion_head_holdout"
    for key in ("data_sha256", "split_sha256", "trajectory_sha256"):
        with pytest.raises(ValueError, match=key):
            validate_checkpoint_split(config, {**split, key: "changed"}, manifest)
    with pytest.raises(ValueError, match="train split"):
        validate_checkpoint_split({"training_metadata": {"train_split": split}}, split, manifest)
    with pytest.raises(ValueError, match="requires the val"):
        validate_checkpoint_split(config, saved, manifest)
    assert validate_checkpoint_split({"training_metadata": {}}, split, manifest)["status"] == "unknown"
    legacy = {"training_metadata": {"train_split": {"split": "train"}, "train_json": manifest}}
    assert validate_checkpoint_split(legacy, split, manifest)["status"] == "unknown"
    with pytest.raises(ValueError, match="data path differs"):
        validate_checkpoint_split(legacy, split, tmp_path / "other.json")


def test_cli_defaults_to_full_val_and_cannot_select_train():
    parser = build_parser()
    args = parser.parse_args(["--checkpoint", "checkpoint"])
    assert args.split == "val" and args.max_samples is None
    assert args.output is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "checkpoint", "--split", "train"])


def test_cli_restores_saved_processors_and_writes_verified_json_before_rejecting_changed_split(tmp_path, monkeypatch):
    import transformers
    from torch.utils.data._utils.collate import default_collate
    from src import eval_diffusion_head as evaluation

    checkpoint = tmp_path / "checkpoint"
    manifest = tmp_path / "data.json"
    output = tmp_path / "report.json"
    split_info = {"split": "val", "data_sha256": "data", "split_sha256": "split",
                  "trajectory_sha256": "traj", "num_samples": 3}
    config = {"head_config": {}, "training_metadata": {"train_split": {**split_info, "split": "train"},
                                    "train_json": str(manifest)}}
    tokenizer = SimpleNamespace(pad_token_id=0, padding_side="left", add_eos_token=True)
    # Give the fake tokenizer a real length, as required by saved embedding checks.
    class Tokenizer(SimpleNamespace):
        def __len__(self):
            return 32
    tokenizer = Tokenizer(**vars(tokenizer))
    processor = object()
    loads = []
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda path, **kwargs: loads.append(("tokenizer", path)) or tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda path, **kwargs: loads.append(("processor", path)) or processor)
    monkeypatch.setattr(evaluation.AeroVLADiffusionModel, "read_checkpoint_config", lambda path: config)
    monkeypatch.setattr(evaluation.AeroVLADiffusionModel, "from_pretrained",
                        lambda path, **kwargs: loads.append(("model", path)) or FixedPolicy())
    monkeypatch.setattr(evaluation, "DiffusionDataCollator", lambda *args, **kwargs: default_collate)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, root, data_json, loaded_tokenizer, loaded_processor, **kwargs):
            assert loaded_tokenizer is tokenizer and loaded_processor is processor
            assert kwargs["split"] == "val"
            self.split_info = dict(split_info)
            self.data = records()

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    monkeypatch.setattr(evaluation, "AeroVLADiffusionDataset", Dataset)
    args = ["--checkpoint", str(checkpoint), "--data-json", str(manifest),
            "--device", "cpu", "--backbone-dtype", "float32", "--num-workers", "0",
            "--output-json", str(output)]
    evaluation.main(args)
    report = json.loads(output.read_text())
    assert report["metrics"]["num_samples"] == 3
    assert report["split_provenance"]["status"] == "verified_diffusion_head_holdout"
    assert loads == [(name, checkpoint) for name in ("tokenizer", "processor", "model")]
    assert tokenizer.padding_side == "right" and tokenizer.add_eos_token is False
    split_info["split_sha256"] = "changed"
    loads.clear()
    with pytest.raises(ValueError, match="split_sha256"):
        evaluation.main(args)
    assert not any(name == "model" for name, _ in loads)
