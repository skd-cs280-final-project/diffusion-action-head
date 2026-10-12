"""CPU tests for diffusion equations, masking, gradients, and reproducibility."""

import io
import json
import unittest
from unittest.mock import patch

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from src.model_wrapper.diffusion_action_head import DiffusionActionHead, DiffusionActionHeadConfig


class DiffusionActionHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(7)
        self.config = DiffusionActionHeadConfig(
            condition_dim=12, hidden_dim=16, num_heads=4, num_layers=1,
            action_horizon=3, num_train_timesteps=10, num_inference_steps=4,
        )
        self.head = DiffusionActionHead(self.config)
        self.condition = torch.randn(2, 5, 12)
        self.actions = torch.rand(2, 3, 4) * 2 - 1
        self.noise = torch.randn_like(self.actions)
        self.timesteps = torch.tensor([1, 7])

    def test_scheduler_uses_cosine_epsilon_with_clipping(self):
        scheduler = self.head.noise_scheduler
        self.assertEqual(scheduler.config.beta_schedule, "squaredcos_cap_v2")
        self.assertEqual(scheduler.config.prediction_type, "epsilon")
        self.assertEqual(scheduler.config.timestep_spacing, "trailing")
        self.assertTrue(scheduler.config.clip_sample)
        alpha = scheduler.alphas_cumprod[self.timesteps][:, None, None]
        expected = alpha.sqrt() * self.actions + (1 - alpha).sqrt() * self.noise
        actual = scheduler.add_noise(self.actions, self.noise, self.timesteps)
        torch.testing.assert_close(actual, expected)

    def test_loss_supplies_noised_targets_and_uses_epsilon(self):
        captured = {}

        def predict(noisy_actions, timesteps, condition, condition_mask):
            captured["noisy_actions"] = noisy_actions
            return self.noise

        with patch.object(self.head, "forward", side_effect=predict):
            loss = self.head.compute_loss(
                self.condition, self.actions, noise=self.noise, timesteps=self.timesteps
            )
        expected = self.head.noise_scheduler.add_noise(self.actions, self.noise, self.timesteps)
        torch.testing.assert_close(captured["noisy_actions"], expected)
        self.assertEqual(loss.item(), 0.0)

    def test_masked_loss_averages_only_valid_coordinates(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        actions = self.actions.clone()
        actions[~mask] = float("nan")
        with patch.object(self.head, "forward", return_value=torch.zeros_like(self.noise)):
            loss = self.head.compute_loss(
                self.condition, actions, action_mask=mask,
                noise=self.noise, timesteps=self.timesteps,
            )
        torch.testing.assert_close(loss, self.noise[mask].square().mean())
        coordinate_mask = mask[..., None].expand_as(self.actions).clone()
        coordinate_mask[:, :, -1] = False
        with patch.object(self.head, "forward", return_value=torch.zeros_like(self.noise)):
            loss = self.head.compute_loss(
                self.condition, self.actions, action_mask=coordinate_mask,
                noise=self.noise, timesteps=self.timesteps,
            )
        torch.testing.assert_close(loss, self.noise[coordinate_mask].square().mean())

    def test_padding_condition_cannot_change_output(self):
        mask = torch.tensor([[True, True, False, False, False], [False, True, True, True, False]])
        padded = self.condition.clone()
        padded[~mask] = float("nan")
        expected = self.head(self.actions, self.timesteps, self.condition, mask)
        actual = self.head(self.actions, self.timesteps, padded, mask)
        torch.testing.assert_close(actual, expected)
        # The timestep memory token also makes an entirely masked condition finite.
        actual = self.head(self.actions, 3, padded, torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(actual).all())

    def test_causal_decoder_does_not_read_future_actions(self):
        altered = self.actions.clone()
        altered[:, 1:] = 100
        before = self.head(self.actions, self.timesteps, self.condition)
        after = self.head(altered, self.timesteps, self.condition)
        torch.testing.assert_close(before[:, 0], after[:, 0])

    def test_loss_backward_and_fixed_batch_optimization(self):
        optimizer = torch.optim.AdamW(self.head.parameters(), lr=0.01)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss = self.head.compute_loss(
                self.condition, self.actions, noise=self.noise, timesteps=self.timesteps,
            )
            loss.backward()
            for name, parameter in self.head.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_sampling_reproducible_bounded_and_preserves_mode(self):
        self.head.train()
        first = self.head.sample(self.condition, generator=torch.Generator().manual_seed(18))
        second = self.head.sample(self.condition, generator=torch.Generator().manual_seed(18))
        self.assertEqual(first.shape, self.actions.shape)
        self.assertFalse(first.requires_grad)
        self.assertTrue(self.head.training)
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(first.abs().max().item(), 1.0)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        third = self.head.sample(self.condition, generator=torch.Generator().manual_seed(19))
        self.assertFalse(torch.equal(first, third))

    def test_mixed_condition_dtype_and_serialization(self):
        actual = self.head(self.actions, 2, self.condition.bfloat16())
        self.assertEqual(actual.dtype, torch.float32)
        config = DiffusionActionHeadConfig.from_dict(json.loads(json.dumps(self.config.to_dict())))
        restored = DiffusionActionHead(config)
        buffer = io.BytesIO()
        torch.save(self.head.state_dict(), buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(actual, restored(self.actions, 2, self.condition.bfloat16()))
        self.assertIn("causal_mask", dict(self.head.named_buffers()))
        self.assertIn("time_embedding.frequencies", dict(self.head.named_buffers()))

    def test_sampling_starts_at_terminal_timestep_and_handles_uneven_spacing(self):
        head = DiffusionActionHead(DiffusionActionHeadConfig(
            condition_dim=12, hidden_dim=16, num_heads=4, num_layers=1,
            num_train_timesteps=100, num_inference_steps=10,
        ))
        for steps in (10, 30, 1):
            scheduler = DDPMScheduler.from_config(head.noise_scheduler.config)
            with patch.object(DDPMScheduler, "from_config", return_value=scheduler):
                actions = head.sample(self.condition, num_inference_steps=steps)
            self.assertTrue(torch.isfinite(actions).all())
            self.assertEqual(scheduler.timesteps[0].item(), 99)
            self.assertEqual(len(scheduler.timesteps), steps)
            for current, following in zip(scheduler.timesteps[:-1], scheduler.timesteps[1:]):
                self.assertEqual(scheduler.previous_timestep(current).item(), following.item())
            self.assertEqual(scheduler.previous_timestep(scheduler.timesteps[-1]).item(), -1)
        self.assertFalse(head.noise_scheduler.custom_timesteps)

    def test_invalid_shapes_masks_and_timesteps_fail_clearly(self):
        with self.assertRaises(ValueError):
            self.head(self.actions[:, :1], 2, self.condition)
        with self.assertRaises(ValueError):
            self.head(self.actions, 10, self.condition)
        with self.assertRaises(ValueError):
            self.head(self.actions, 1.5, self.condition)
        with self.assertRaises(ValueError):
            self.head(self.actions, 1, self.condition, torch.ones(2, 6))
        with self.assertRaises(ValueError):
            self.head.compute_loss(self.condition, self.actions, action_mask=torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            self.head.sample(self.condition, num_inference_steps=11)
        with self.assertRaises(ValueError):
            self.head.compute_loss(self.condition, self.actions * float("nan"))
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.head.compute_loss(self.condition, torch.full_like(self.actions, 2.0))

    def test_config_validation_and_single_step_horizon(self):
        with self.assertRaises(ValueError):
            DiffusionActionHeadConfig(condition_dim=12, hidden_dim=15, num_heads=4)
        with self.assertRaises(ValueError):
            DiffusionActionHeadConfig(condition_dim=12, action_horizon=0)
        with self.assertRaises(ValueError):
            DiffusionActionHeadConfig(condition_dim=12, num_train_timesteps=2, num_inference_steps=3)
        head = DiffusionActionHead(DiffusionActionHeadConfig(
            condition_dim=12, hidden_dim=15, num_heads=3, num_layers=1,
            num_train_timesteps=2, num_inference_steps=2,
        ))
        output = head.sample(self.condition)
        self.assertEqual(output.shape, (2, 1, 4))


if __name__ == "__main__":
    unittest.main()
