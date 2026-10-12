"""Data-contract tests; run with ``python -m unittest discover -s tests``."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

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
