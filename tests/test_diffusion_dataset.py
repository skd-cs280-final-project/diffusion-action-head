"""Data-contract tests; run with ``python -m unittest discover -s tests``."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_dataset import (  # noqa: E402
    ACTION_MAX,
    ACTION_MIN,
    AeroVLADiffusionDataset,
    DiffusionDataCollator,
    denormalize_actions,
    normalize_actions,
)
from diffusion_split import deterministic_subset
from uav_history import STATE_DIM, STATE_VALUE_DIM, encode_state_history


class RecordingProcessor:
    def __init__(self, mapping=False):
        self.mapping = mapping
        self.seen = None

    def image_processor(self, images, return_tensors):
        assert return_tensors == "pt"
        self.seen = images.copy()
        pixels = torch.ones(1, 6, 8, 8)
        return {"pixel_values": {"tower": pixels} if self.mapping else pixels}


class PromptTokenizer:
    pad_token_id = 0
    add_eos_token = False

    def __init__(self, padding_side="right"):
        self.padding_side = padding_side
        self.texts = None

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is True
        self.texts = texts
        # BOS plus a deterministic character encoding, without a target EOS.
        rows = [[1] + [ord(char) + 3 for char in text] for text in texts]
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            start = 0 if self.padding_side == "right" else length - len(row)
            ids[i, start:start + len(row)] = torch.tensor(row)
            mask[i, start:start + len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class NormalizationTests(unittest.TestCase):
    def test_endpoints_midpoints_and_roundtrip(self):
        physical = torch.tensor([ACTION_MIN, [2.5, 0.0, 0.0, 0.5], ACTION_MAX])
        saved = physical.clone()
        normalized = normalize_actions(physical)
        torch.testing.assert_close(normalized, torch.tensor([[-1.] * 4, [0.] * 4, [1.] * 4]))
        torch.testing.assert_close(denormalize_actions(normalized), physical)
        torch.testing.assert_close(physical, saved)
        self.assertEqual(normalized.dtype, torch.float32)

    def test_clip_matches_original_physical_action_ranges(self):
        normalized = normalize_actions([[-8.0, -9.0, -2.0, 0.0], [8.0, 9.0, 2.0, 1.0]])
        torch.testing.assert_close(normalized, torch.tensor([[-1.] * 4, [1.] * 4]))
        torch.testing.assert_close(denormalize_actions(normalized * 3), torch.tensor([ACTION_MIN, ACTION_MAX]))

    def test_invalid_action_shape_and_nonfinite_values_fail(self):
        for value in (1.0, [1., 2., 3.], [0., float("nan"), 0., 0.], [0., float("inf"), 0., 0.]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_actions(value)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for camera, color in (("frontcamera", (255, 0, 0)), ("downcamera", (0, 0, 255))):
            folder = self.root / "map" / "trajectory" / camera
            folder.mkdir(parents=True)
            Image.new("RGB", (13, 17), color).save(folder / "000000.png")
        self.sample = {
            "traj_rel_dir": "map/trajectory",
            "img_name": "000000.png",
            "instruction": "  Fly forward  ",
            "label": {"fwd": 2.5, "down": 0.0, "yaw": 0.0},
            "is_last_step": False,
            "is_penultimate": False,
        }
        self.tokenizer = PromptTokenizer()

    def dataset(self, samples=None, processor=None, **kwargs):
        index = self.root / "train.json"
        index.write_text(json.dumps([self.sample] if samples is None else samples), encoding="utf-8")
        return AeroVLADiffusionDataset(self.root, index, self.tokenizer, processor or RecordingProcessor(), **kwargs)

    def history_sample(self, frame, trajectory="map/trajectory", position=None, write_log=True):
        sample = dict(self.sample, traj_rel_dir=trajectory, img_name=f"{frame}.png")
        for camera in ("frontcamera", "downcamera"):
            folder = self.root / trajectory / camera
            folder.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (13, 17)).save(folder / sample["img_name"])
        if write_log:
            folder = self.root / trajectory / "log"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{frame:06d}.json").write_text(json.dumps({"sensors": {"state": {
                "position": [frame, 0, 0] if position is None else position,
                "orientation": [0, 0, 0, 1],
            }}}), encoding="utf-8")
        return sample

    def test_history_uses_numeric_frame_order_and_never_future_or_other_trajectory(self):
        current = self.history_sample(10)
        future = self.history_sample(11, write_log=False)
        past = self.history_sample(2)
        other = self.history_sample(9, trajectory="map/other", position=[999, 0, 0])
        dataset = self.dataset([current, future, other, past], history_length=3)
        history = dataset[0]["state_history"]
        self.assertEqual(dataset[0]["state_history_mask"].tolist(), [False, True, True])
        torch.testing.assert_close(history[1:, 0], torch.tensor([-.4, 0.]))
        self.assertFalse(history[:, 9:13].any())
        self.assertFalse(history[:, STATE_VALUE_DIM + 9:].any())

    def test_frame_subsampling_preserves_history_from_full_selected_dataset(self):
        samples = [self.history_sample(frame) for frame in range(8)]
        dataset = self.dataset(samples, history_length=4)
        subset = deterministic_subset(dataset, max_samples=1, seed=0)
        selected_frame = subset.indices[0]
        self.assertGreaterEqual(selected_frame, 3)
        feature = subset[0]
        self.assertTrue(feature["state_history_mask"].all())
        torch.testing.assert_close(feature["state_history"][:, 0], torch.tensor([-.15, -.1, -.05, 0.]))

    def test_history_does_not_use_expert_labels_or_merged_future_state(self):
        samples = [self.history_sample(frame) for frame in (0, 1)]
        dataset = self.dataset(samples, history_length=2)
        before = dataset[1]["state_history"].clone()
        samples[0]["label"] = {"fwd": 999, "down": 999, "yaw": 999}
        samples[0]["trajectory"] = [{"position": [999, 999, 999]}]
        dataset = self.dataset(samples, history_length=2)
        torch.testing.assert_close(dataset[1]["state_history"], before)

    def test_history_missing_logs_and_invalid_pose_fail_without_blank_history(self):
        sample = self.history_sample(1, write_log=False)
        with self.assertRaises(FileNotFoundError):
            self.dataset([sample], history_length=2)[0]
        sample = self.history_sample(2, position=None)
        path = self.root / "map/trajectory/log/000002.json"
        path.write_text(json.dumps({"sensors": {"state": {"position": [0, 0, 0]}}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "orientation"):
            self.dataset([sample], history_length=2)[0]

    def test_history_rejects_log_frame_misalignment(self):
        sample = self.history_sample(3)
        path = self.root / "map/trajectory/log/000003.json"
        record = json.loads(path.read_text())
        record["frame"] = 4
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatched frame"):
            self.dataset([sample], history_length=2)[0]

    def test_offline_and_online_refill_after_duplicate_timestamps(self):
        samples = [self.history_sample(frame) for frame in range(5)]
        records = []
        for frame, timestamp in enumerate((1, 2, 3, 3, 3)):
            path = self.root / "map/trajectory/log" / f"{frame:06d}.json"
            record = json.loads(path.read_text())
            record["frame"] = frame
            record["sensors"]["state"]["timestamp"] = timestamp * 1_000_000_000
            record["sensors"]["state"]["position"] = [timestamp, 0, 0]
            record["sensors"]["state"]["linear_velocity"] = [1., 2., 3.]
            path.write_text(json.dumps(record), encoding="utf-8")
            records.append(record)
        feature = self.dataset(samples, history_length=3)[-1]
        online_history, online_mask = encode_state_history(records, 3)
        torch.testing.assert_close(feature["state_history"], online_history)
        torch.testing.assert_close(feature["state_history_mask"], online_mask)
        self.assertTrue(online_mask.all())
        self.assertTrue(feature["state_history"][:, STATE_VALUE_DIM + 9:].bool().all())

    def test_history_split_is_applied_before_trajectory_index(self):
        samples = [self.history_sample(0), self.history_sample(0, trajectory="map/val")]
        selected_info = {"num_samples": 1, "num_trajectories": 1}
        with patch("diffusion_dataset.select_trajectory_split", return_value=([samples[0]], selected_info)):
            dataset = self.dataset(samples, history_length=4, split_json="unused.json", split="train")
        self.assertEqual(set(dataset._history_frames), {"map/trajectory"})

    def test_collator_stacks_state_history_without_casting_to_image_precision(self):
        dataset = self.dataset([self.history_sample(0), self.history_sample(1)], history_length=4)
        batch = DiffusionDataCollator(self.tokenizer, pixel_dtype=torch.bfloat16)([dataset[0], dataset[1]])
        self.assertEqual(batch["state_history"].shape, (2, 4, STATE_DIM))
        self.assertEqual(batch["state_history"].dtype, torch.float32)
        self.assertEqual(batch["state_history_mask"].dtype, torch.bool)
        self.assertEqual(batch["state_history_mask"].sum(dim=1).tolist(), [1, 2])
        self.assertEqual(self.tokenizer.texts, ["Fly forward\nAction: "] * 2)

    def test_collator_rejects_mixed_history_or_invalid_current_mask(self):
        dataset = self.dataset([self.history_sample(0)], history_length=2)
        feature = dataset[0]
        incomplete = {key: value for key, value in feature.items() if key != "state_history_mask"}
        collator = DiffusionDataCollator(self.tokenizer)
        with self.assertRaisesRegex(ValueError, "both"):
            collator([feature, incomplete])
        with self.assertRaisesRegex(ValueError, "current"):
            collator([dict(feature, state_history_mask=torch.tensor([True, False]))])

    def test_history_index_rejects_ambiguous_frames_and_invalid_length(self):
        for length in (-1, True, 1.5):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "history_length"):
                self.dataset(history_length=length)
        with self.assertRaisesRegex(ValueError, "Duplicate history"):
            self.dataset([self.sample, self.sample], history_length=2)
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.dataset([dict(self.sample, img_name="not_a_frame.png")], history_length=2)

    def test_preserve_image_mosaic_and_prompt_without_teacher_forcing(self):
        processor = RecordingProcessor()
        feature = self.dataset(processor=processor)[0]
        self.assertEqual(set(feature), {"image", "prompt_only", "actions"})
        self.assertEqual(feature["prompt_only"], "Fly forward\nAction: ")
        self.assertEqual(feature["image"].shape, (6, 8, 8))
        torch.testing.assert_close(feature["actions"], torch.tensor([[0., 0., 0., -1.]]))
        self.assertEqual(processor.seen.size, (224, 448))
        self.assertEqual(processor.seen.getpixel((100, 100)), (255, 0, 0))
        self.assertEqual(processor.seen.getpixel((100, 300)), (0, 0, 255))

    def test_last_and_penultimate_steps_preserve_land_rule(self):
        for last, penultimate, expected in ((False, False, -1), (True, False, 1), (False, True, 1), (True, True, 1)):
            sample = dict(self.sample, is_last_step=last, is_penultimate=penultimate)
            with self.subTest(last=last, penultimate=penultimate):
                self.assertEqual(self.dataset([sample])[0]["actions"][0, 3], expected)

    def test_changing_target_does_not_change_model_inputs(self):
        second = dict(self.sample, label={"fwd": 4.1, "down": -2.2, "yaw": 0.76}, is_last_step=True)
        dataset = self.dataset([self.sample, second])
        batch = DiffusionDataCollator(self.tokenizer)([dataset[0], dataset[1]])
        self.assertEqual(set(batch), {"input_ids", "attention_mask", "pixel_values", "actions"})
        torch.testing.assert_close(batch["input_ids"][0], batch["input_ids"][1])
        torch.testing.assert_close(batch["pixel_values"][0], batch["pixel_values"][1])
        self.assertFalse(torch.equal(batch["actions"][0], batch["actions"][1]))

    def test_horizons_without_temporal_metadata_fail(self):
        with self.assertRaisesRegex(ValueError, "action_horizon=1"):
            self.dataset(action_horizon=2)

    def test_invalid_json_and_flags_fail(self):
        with self.assertRaisesRegex(ValueError, "list"):
            self.dataset(samples={"records": []})
        for sample in (dict(self.sample, is_last_step="False"), dict(self.sample, instruction="  ")):
            with self.subTest(sample=sample), self.assertRaisesRegex(ValueError, "index 0"):
                self.dataset([sample])[0]

    def test_missing_image_is_not_replaced_by_blank_observation(self):
        self.sample["img_name"] = "missing.png"
        with self.assertRaises(FileNotFoundError):
            self.dataset()[0]

    def test_collator_padding_masks_target_shape_and_image_dtype(self):
        dataset = self.dataset([self.sample, dict(self.sample, instruction="Go")])
        for side in ("left", "right"):
            tokenizer = PromptTokenizer(side)
            batch = DiffusionDataCollator(tokenizer, pixel_dtype=torch.bfloat16)([dataset[0], dataset[1]])
            with self.subTest(padding=side):
                self.assertEqual(batch["actions"].shape, (2, 1, 4))
                self.assertEqual(batch["actions"].dtype, torch.float32)
                self.assertEqual(batch["pixel_values"].dtype, torch.bfloat16)
                self.assertEqual(batch["pixel_values"].shape, (2, 6, 8, 8))
                self.assertTrue(torch.equal(batch["attention_mask"], batch["input_ids"].ne(0).long()))

    def test_mapping_image_processors_are_batched(self):
        dataset = self.dataset(processor=RecordingProcessor(mapping=True))
        batch = DiffusionDataCollator(self.tokenizer)([dataset[0], dataset[0]])
        self.assertEqual(batch["pixel_values"]["tower"].shape, (2, 6, 8, 8))

    def test_prompt_overflow_is_rejected_without_truncation(self):
        feature = self.dataset()[0]
        with self.assertRaisesRegex(ValueError, "exceeding max_length"):
            DiffusionDataCollator(self.tokenizer, max_length=3)([feature])
        self.assertEqual(self.tokenizer.texts, ["Fly forward\nAction: "])

    def test_collator_rejects_eos_empty_batch_and_invalid_targets(self):
        tokenizer = PromptTokenizer()
        tokenizer.add_eos_token = True
        with self.assertRaisesRegex(ValueError, "add_eos_token"):
            DiffusionDataCollator(tokenizer)
        collator = DiffusionDataCollator(self.tokenizer)
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collator([])
        feature = self.dataset()[0]
        for target in (torch.ones(2, 4), torch.ones(1, 4) * 1.2):
            with self.subTest(target=target), self.assertRaises(ValueError):
                collator([dict(feature, actions=target)])


if __name__ == "__main__":
    unittest.main()
