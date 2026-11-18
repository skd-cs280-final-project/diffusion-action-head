"""Geometry, causality and missing-data contracts for shared UAV histories."""

import copy
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uav_history import HISTORY_SCHEMA, STATE_DIM, STATE_VALUE_DIM, encode_state_history


def observation(position=(0, 0, 0), yaw=0, **extra):
    return {"sensors": {"state": {
        "position": list(position),
        "orientation": [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)],
        **extra,
    }}}


class StateHistoryTests(unittest.TestCase):
    def test_padding_current_pose_and_missing_motion_have_separate_masks(self):
        history, mask = encode_state_history([observation()], 4)
        self.assertEqual(history.shape, (4, STATE_DIM))
        self.assertEqual(history.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [False, False, False, True])
        self.assertFalse(history[:3].any())
        torch.testing.assert_close(history[-1, :9], torch.tensor([0., 0., 0., 0., 1., 0., 1., 0., 1.]))
        self.assertEqual(history[-1, STATE_VALUE_DIM:].tolist(), [1.] * 9 + [0.] * 4)
        self.assertFalse(history[-1, 9:13].any())
        self.assertEqual(HISTORY_SCHEMA["state_dim"], STATE_DIM)

    def test_current_yaw_frame_rotates_all_positions_and_world_velocity(self):
        past = observation((0, 10, -4), linear_velocity=[0, 5, -2])
        current = observation((0, 20, -2), yaw=math.pi / 2, linear_velocity=[0, 0, 0])
        history, mask = encode_state_history([past, current], 2)
        self.assertTrue(mask.all())
        torch.testing.assert_close(history[0, :3], torch.tensor([-.5, 0., -.1]), atol=1e-7, rtol=0)
        torch.testing.assert_close(history[0, 7:9], torch.tensor([-1., 0.]), atol=1e-7, rtol=0)
        torch.testing.assert_close(history[0, 9:12], torch.tensor([1., 0., -.4]), atol=1e-7, rtol=0)
        self.assertTrue(history[:, STATE_VALUE_DIM + 9:STATE_VALUE_DIM + 12].bool().all())

    def test_roll_pitch_are_retained_and_quaternion_is_normalized(self):
        state = observation()["sensors"]["state"]
        state["orientation"] = [math.sin(math.pi / 8) * 3, 0, 0, math.cos(math.pi / 8) * 3]
        history, _ = encode_state_history([state], 1)
        torch.testing.assert_close(history[0, 3:7], torch.tensor([2**-.5, 2**-.5, 0., 1.]))

    def test_positive_airsim_nanoseconds_control_age_and_remove_copies(self):
        stamp = 1_750_000_000_000_000_000
        first = observation(timestamp=stamp)
        second = observation((5, 0, 0), timestamp=stamp + 2_000_000_000)
        history, mask = encode_state_history([first] + [copy.deepcopy(second) for _ in range(5)], 3)
        self.assertEqual(mask.tolist(), [False, True, True])
        torch.testing.assert_close(history[1:, 12], torch.tensor([.2, 0.]))
        self.assertTrue(history[1:, STATE_VALUE_DIM + 12].bool().all())

    def test_nanosecond_integer_subtraction_preserves_sub_microsecond_intervals(self):
        stamp = 1_750_000_000_000_000_000
        history, _ = encode_state_history([observation(timestamp=stamp), observation(timestamp=stamp + 100)], 2)
        self.assertAlmostEqual(history[0, 12].item(), 1e-8, places=14)

    def test_new_timestamps_preserve_real_hover_observations(self):
        rows = [observation(timestamp=(i + 1) * 1_000_000_000) for i in range(4)]
        history, mask = encode_state_history(rows, 4)
        self.assertTrue(mask.all())
        torch.testing.assert_close(history[:, 12], torch.tensor([.3, .2, .1, 0.]))

    def test_frame_identity_and_object_identity_deduplicate_without_pose_heuristics(self):
        first = observation()
        second = copy.deepcopy(first)
        _, mask = encode_state_history([first, first, second], 4)
        self.assertEqual(mask.tolist(), [False, False, True, True])
        first["frame_id"], second["frame_id"] = 3, 3
        _, mask = encode_state_history([first, second], 4)
        self.assertEqual(mask.tolist(), [False, False, False, True])
        second["frame_id"] = 4
        _, mask = encode_state_history([first, second], 4)
        self.assertEqual(mask.tolist(), [False, False, True, True])

    def test_missing_current_timestamp_masks_all_ages_and_never_invents_velocity(self):
        history, _ = encode_state_history([observation(timestamp=1_000_000_000), observation((4, 0, 0))], 2)
        self.assertFalse(history[:, 9:13].any())
        self.assertFalse(history[:, STATE_VALUE_DIM + 9:].any())

    def test_partial_invalid_velocity_is_masked_after_rotation(self):
        history, _ = encode_state_history([observation(linear_velocity=[float("nan"), 1, -2])], 1)
        torch.testing.assert_close(history[0, 9:12], torch.tensor([0., 0., -.4]))
        self.assertEqual(history[0, STATE_VALUE_DIM + 9:STATE_VALUE_DIM + 12].tolist(), [0., 0., 1.])

    def test_unordered_timestamps_fail_instead_of_negative_age(self):
        with self.assertRaisesRegex(ValueError, "later than|ordered"):
            encode_state_history([observation(timestamp=5e9), observation(timestamp=1e9)], 2)

    def test_missing_or_malformed_required_pose_fails(self):
        for name, value in (("position", None), ("position", [0, float("nan"), 0]),
                            ("orientation", [0, 0, 0, 0]), ("orientation", [0, 0, 0])):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                row = observation()
                row["sensors"]["state"][name] = value
                encode_state_history([row], 1)
        with self.assertRaisesRegex(ValueError, "at least"):
            encode_state_history([], 4)

    def test_clips_normalization_and_keeps_latest_window(self):
        rows = [observation((x, 0, 0), timestamp=(x + 1) * 1_000_000_000, linear_velocity=[100, 0, -100])
                for x in (0, 100, 200)]
        history, mask = encode_state_history(rows, 2)
        self.assertTrue(mask.all())
        self.assertEqual(history[:, 0].tolist(), [-1., 0.])
        self.assertEqual(history[:, 12].tolist(), [1., 0.])
        self.assertLessEqual(history.abs().max().item(), 1.)

    def test_history_length_contract(self):
        history, mask = encode_state_history([], 0)
        self.assertEqual(history.shape, (0, STATE_DIM))
        self.assertEqual(mask.shape, (0,))
        for length in (-1, True, 2.0):
            with self.subTest(length=length), self.assertRaises(ValueError):
                encode_state_history([observation()], length)


if __name__ == "__main__":
    unittest.main()
