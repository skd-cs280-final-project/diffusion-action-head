"""Training-loop regressions independent of the 7B checkpoint."""
import copy
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.train_diffusion_head import assert_only_head_trainable, train_head


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.9))
        self.diffusion_head = nn.Linear(4, 4)
        self.backbone.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, observations, actions):
        with torch.no_grad():
            condition = self.backbone(observations)
        return (self.diffusion_head(condition) - actions).square().mean()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.data = [{"observations": torch.randn(3), "actions": torch.randn(4)} for _ in range(5)]

    def test_uneven_microbatches_match_full_batch_and_backbone_unchanged(self):
        accumulated = TinyPolicy()
        whole = copy.deepcopy(accumulated)
        before_backbone = copy.deepcopy(accumulated.backbone.state_dict())
        before_head = copy.deepcopy(accumulated.diffusion_head.state_dict())
        kwargs = dict(device="cpu", epochs=1, learning_rate=0.01, weight_decay=0,
                      max_grad_norm=100.0, warmup_ratio=0)
        metrics = []
        train_head(accumulated, DataLoader(self.data, batch_size=2),
                   gradient_accumulation_steps=3, log_callback=metrics.append, **kwargs)
        train_head(whole, DataLoader(self.data, batch_size=5), gradient_accumulation_steps=1, **kwargs)
        self.assertEqual(metrics[0]["samples"], 5)
        for key, value in accumulated.diffusion_head.state_dict().items():
            torch.testing.assert_close(value, whole.diffusion_head.state_dict()[key])
        self.assertTrue(any(not torch.equal(value, before_head[key])
                            for key, value in accumulated.diffusion_head.state_dict().items()))
        for key, value in accumulated.backbone.state_dict().items():
            self.assertTrue(torch.equal(value, before_backbone[key]))
        self.assertFalse(accumulated.backbone.training)
        self.assertTrue(all(parameter.grad is None for parameter in accumulated.backbone.parameters()))

    def test_tail_updates_and_max_steps(self):
        metrics, saves = [], []
        steps = train_head(TinyPolicy(), DataLoader(self.data, batch_size=2), device="cpu",
                           epochs=3, gradient_accumulation_steps=2, max_steps=3,
                           log_callback=metrics.append, save_callback=saves.append, save_steps=2)
        self.assertEqual(steps, 3)
        self.assertEqual([metric["samples"] for metric in metrics], [4, 1, 4])
        self.assertEqual(saves, [2])

    def test_unfrozen_parameter_rejected(self):
        model = TinyPolicy()
        model.backbone[0].weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Only diffusion_head"):
            assert_only_head_trainable(model)

    def test_nonfinite_loss_rejected(self):
        self.data[0]["actions"][0] = float("nan")
        with self.assertRaises(FloatingPointError):
            train_head(TinyPolicy(), DataLoader(self.data, batch_size=5), device="cpu", epochs=1)

    def test_validation_at_epoch_and_final_step_without_duplicate_and_restores_training(self):
        model = TinyPolicy()
        calls = []

        def validate(step, epoch):
            calls.append((step, epoch))
            self.assertTrue(model.training)
            self.assertFalse(model.backbone.training)
            model.eval()

        train_head(model, DataLoader(self.data, batch_size=2), device="cpu", epochs=3,
                   gradient_accumulation_steps=2, max_steps=3, eval_steps=2,
                   validation_callback=validate)
        self.assertEqual(calls, [(2, 1), (3, 2)])
        self.assertTrue(model.training)
        self.assertFalse(model.backbone.training)

    def test_validation_is_on_optimizer_steps_after_accumulation(self):
        calls = []
        train_head(TinyPolicy(), DataLoader(self.data, batch_size=1), device="cpu", epochs=1,
                   gradient_accumulation_steps=2, eval_steps=1,
                   validation_callback=lambda step, epoch: calls.append((step, epoch)))
        self.assertEqual(calls, [(1, 1), (2, 1), (3, 1)])


class InferenceConversionTests(unittest.TestCase):
    def test_diffusion_evaluation_returns_continuous_actions_and_land(self):
        from src.model_wrapper.aerovla_wrapper_ui import AerialVLAWrapper

        class FixedPolicy:
            def predict_action(self, **inputs):
                return torch.tensor([[[-1.0, 0.0, 1.0, -1.0]], [[1.0, -1.0, 0.0, 1.0]]])

        wrapper = AerialVLAWrapper.__new__(AerialVLAWrapper)
        wrapper.diffusion_mode = True
        wrapper.model = FixedPolicy()
        actions, stops = wrapper.run({}, None, None)
        self.assertEqual(stops, [False, True])
        self.assertEqual(actions[0]["fwd"], 0.0)
        self.assertAlmostEqual(actions[0]["yaw"], 1.1, places=6)
        self.assertEqual(actions[1], {"fwd": 5.0, "down": -5.0, "yaw": 0.0})


if __name__ == "__main__":
    unittest.main()
