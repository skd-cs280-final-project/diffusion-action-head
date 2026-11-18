"""Joint-training invariants without loading the 7B model or using a GPU."""

import copy
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train_diffusion_joint import parameter_groups, train_joint


GROUP_NAMES = ("vision_lora", "llm_lora", "projector", "diffusion_head")


class TinyJointPolicy(nn.Module):
    """Independent differentiable branches expose all four optimizer groups."""

    def __init__(self, detached_group=None):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.original_base = nn.Linear(3, 3, bias=False)
        self.backbone.vision_backbone = nn.Module()
        self.backbone.vision_backbone.lora_A = nn.Linear(3, 4, bias=False)
        self.backbone.language_model = nn.Module()
        self.backbone.language_model.lora_B = nn.Linear(3, 4, bias=False)
        self.backbone.projector = nn.Linear(3, 4, bias=False)
        self.diffusion_head = nn.Linear(3, 4, bias=False)
        self.detached_group = detached_group
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                parameter.fill_(0.05 * (index + 1))
        self.backbone.original_base.requires_grad_(False)

    def forward(self, observations, actions):
        features = self.backbone.original_base(observations)
        branches = {
            "vision_lora": self.backbone.vision_backbone.lora_A(features),
            "llm_lora": self.backbone.language_model.lora_B(features),
            "projector": self.backbone.projector(features),
            "diffusion_head": self.diffusion_head(features),
        }
        if self.detached_group is not None:
            branches[self.detached_group] = branches[self.detached_group].detach()
        return (sum(branches.values()) - actions).square().mean()


@pytest.fixture
def samples():
    return [
        {
            "observations": torch.tensor([1.0 + index, 0.5 + index / 3, 1.0]),
            "actions": torch.tensor([-0.1, 0.2, -0.3, 0.1]) + index / 20,
        }
        for index in range(5)
    ]


def optimizer_groups(model, **overrides):
    arguments = dict(lora_lr=0.01, projector_lr=0.02, head_lr=0.03, weight_decay=0)
    arguments.update(overrides)
    return parameter_groups(model, **arguments)


def run_training(model, loader, **overrides):
    arguments = dict(
        device="cpu", epochs=1, lora_lr=0.01, projector_lr=0.02, head_lr=0.03,
        weight_decay=0, max_grad_norm=100.0, warmup_ratio=0,
    )
    arguments.update(overrides)
    return train_joint(model, loader, **arguments)


def test_optimizer_covers_each_trainable_parameter_once_with_requested_rates():
    model = TinyJointPolicy()
    groups = optimizer_groups(model)
    assert [group["name"] for group in groups] == list(GROUP_NAMES)
    assert [group["lr"] for group in groups] == [0.01, 0.01, 0.02, 0.03]
    actual = [id(parameter) for group in groups for parameter in group["params"]]
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert len(actual) == len(set(actual))
    assert set(actual) == expected
    assert all(parameter.dtype == torch.float32 for group in groups for parameter in group["params"])


def test_uneven_microbatches_equal_whole_batch_and_only_update_allowed_parameters(samples):
    accumulated = TinyJointPolicy()
    whole_batch = copy.deepcopy(accumulated)
    before = {name: parameter.detach().clone() for name, parameter in accumulated.named_parameters()}
    logs, whole_logs = [], []
    steps = run_training(
        accumulated, DataLoader(samples, batch_size=2),
        gradient_accumulation_steps=3, log_callback=logs.append,
    )
    run_training(
        whole_batch, DataLoader(samples, batch_size=5),
        gradient_accumulation_steps=1, log_callback=whole_logs.append,
    )
    assert steps == 1
    assert logs[0]["samples"] == 5
    assert logs[0]["first_step_parameter_updates"] == dict.fromkeys(GROUP_NAMES, True)
    assert all(math.isfinite(value) and value > 0 for value in logs[0]["group_grad_norm"].values())
    # Adam's first update can match even when gradient magnitudes are wrong;
    # compare the actual pre-clipping norms and sample-weighted objective too.
    assert logs[0]["loss"] == pytest.approx(whole_logs[0]["loss"], rel=1e-6)
    assert logs[0]["grad_norm"] == pytest.approx(whole_logs[0]["grad_norm"], rel=1e-6)
    assert logs[0]["group_grad_norm"] == pytest.approx(whole_logs[0]["group_grad_norm"], rel=1e-6)
    for name, parameter in accumulated.named_parameters():
        torch.testing.assert_close(parameter, dict(whole_batch.named_parameters())[name])
        if parameter.requires_grad:
            assert not torch.equal(parameter, before[name]), name
        else:
            assert torch.equal(parameter, before[name]), name
            assert parameter.grad is None


def test_tail_updates_max_steps_callbacks_and_training_mode(samples):
    model = TinyJointPolicy()
    logs, saves, validations = [], [], []

    def validate(step, epoch):
        assert model.training
        validations.append((step, epoch))
        model.eval()

    steps = run_training(
        model, DataLoader(samples, batch_size=2), epochs=3,
        gradient_accumulation_steps=2, max_steps=3, log_callback=logs.append,
        save_callback=saves.append, save_steps=2,
        validation_callback=validate, eval_steps=2,
    )
    assert steps == 3
    assert [item["samples"] for item in logs] == [4, 1, 4]
    assert [item["step"] for item in logs] == [1, 2, 3]
    assert [item["epoch"] for item in logs] == [1, 1, 2]
    assert all(item["total_steps"] == 3 for item in logs)
    assert saves == [2]
    assert validations == [(2, 1), (3, 2)]
    assert model.training


@pytest.mark.parametrize("eval_steps,expected", [
    (0, [(3, 1)]),
    (1, [(1, 1), (2, 1), (3, 1)]),
])
def test_validation_intervals_count_optimizer_updates(samples, eval_steps, expected):
    validations = []
    steps = run_training(
        TinyJointPolicy(), DataLoader(samples, batch_size=1),
        gradient_accumulation_steps=2, eval_steps=eval_steps,
        validation_callback=lambda step, epoch: validations.append((step, epoch)),
    )
    assert steps == 3
    assert validations == expected


@pytest.mark.parametrize("name", GROUP_NAMES)
def test_disconnected_group_is_rejected_before_any_optimizer_update(samples, name):
    model = TinyJointPolicy(detached_group=name)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match=f"No finite nonzero first-step gradient for {name}"):
        run_training(model, DataLoader(samples, batch_size=5), gradient_accumulation_steps=1)
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key


@pytest.mark.parametrize("name", GROUP_NAMES)
def test_missing_trainable_group_is_rejected(name):
    model = TinyJointPolicy()
    for group in optimizer_groups(model):
        if group["name"] == name:
            for parameter in group["params"]:
                parameter.requires_grad_(False)
    with pytest.raises(RuntimeError, match="all four parameter groups"):
        optimizer_groups(model)


def test_unfrozen_base_parameter_is_rejected():
    model = TinyJointPolicy()
    model.backbone.original_base.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="Unexpected trainable parameter.*original_base"):
        optimizer_groups(model)


def test_non_float32_trainable_parameter_is_rejected():
    model = TinyJointPolicy()
    model.diffusion_head.double()
    with pytest.raises(RuntimeError, match="must use float32.*diffusion_head"):
        optimizer_groups(model)


@pytest.mark.parametrize("name", ["lora_lr", "projector_lr", "head_lr"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_learning_rate_is_rejected(name, value):
    with pytest.raises(ValueError, match="learning rates"):
        optimizer_groups(TinyJointPolicy(), **{name: value})


@pytest.mark.parametrize("arguments", [
    {"epochs": 0},
    {"gradient_accumulation_steps": 0},
    {"max_steps": 0},
    {"weight_decay": -0.1},
    {"weight_decay": float("nan")},
    {"weight_decay": float("inf")},
    {"max_grad_norm": 0},
    {"max_grad_norm": float("nan")},
    {"max_grad_norm": float("inf")},
    {"warmup_ratio": -0.1},
    {"warmup_ratio": 1},
    {"warmup_ratio": float("nan")},
    {"save_steps": 0},
    {"eval_steps": -1},
])
def test_invalid_training_arguments_are_rejected(samples, arguments):
    with pytest.raises(ValueError):
        run_training(TinyJointPolicy(), DataLoader(samples, batch_size=5), **arguments)


def test_empty_loader_is_rejected():
    with pytest.raises(ValueError, match="dataloader length"):
        run_training(TinyJointPolicy(), DataLoader([], batch_size=1))


def test_nonfinite_loss_is_rejected(samples):
    samples[0]["actions"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="loss"):
        run_training(TinyJointPolicy(), DataLoader(samples, batch_size=5))
