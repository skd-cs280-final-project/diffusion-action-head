"""Static audit tests: real target contract, global indices and fail-closed provenance."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from audit_diffusion_supervision import (  # noqa: E402
    AuditValidationError,
    audit,
    group_statistics,
    normalize_actions,
    select_examples,
    select_trajectory_split,
    target_arrays,
    validate_targets,
)


def sample(trajectory, image, label=(2.5, 0.0, 0.0), last=False, penultimate=False):
    return {"traj_rel_dir": trajectory, "img_name": image, "instruction": "  Fly forward  ",
            "label": dict(zip(("fwd", "down", "yaw"), label)),
            "is_last_step": last, "is_penultimate": penultimate}


class SupervisionTargetTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            sample("map/a", "0.png", (-1.0, 6.0, -2.0)),
            sample("map/b", "0.png", (7.0, -6.0, 2.0), last=True),
            sample("map/a", "1.png", penultimate=True),
            sample("map/b", "1.png", (0.0, -5.0, -1.1), last=True, penultimate=True),
        ]

    def test_float32_normalization_matches_training_including_land_flags_and_bounds(self):
        raw, flags = validate_targets(self.samples)
        clipped, normalized, below, above = target_arrays(raw)
        expected = normalize_actions([[-1, 6, -2, 0], [7, -6, 2, 1], [2.5, 0, 0, 1], [0, -5, -1.1, 1]])
        np.testing.assert_array_equal(normalized, expected.numpy())
        np.testing.assert_array_equal(clipped[:2], np.asarray([[0, 5, -1.1, 0], [5, -5, 1.1, 1]], dtype=np.float32))
        self.assertEqual(normalized.dtype, np.float32)
        self.assertFalse((below[3] | above[3]).any(), "Exact declared endpoints must not count as clipping")

    def test_statistics_are_population_values_and_flags_do_not_double_count_land(self):
        raw, flags = validate_targets(self.samples)
        result = group_statistics(range(4), raw, *target_arrays(raw), flags)
        self.assertEqual(result["raw_4d"]["fwd"]["mean"], 2.125)
        self.assertAlmostEqual(result["raw_4d"]["fwd"]["std_population"], np.std([-1, 7, 2.5, 0]))
        self.assertEqual(result["clipping"]["fwd"], {"below_min": 1, "above_max": 1, "count": 2, "ratio": 0.5})
        self.assertEqual(result["any_axis_clipped"], 2)
        self.assertEqual(result["land"], {"positive": 3, "negative": 1, "positive_ratio": 0.75,
                                         "last": 2, "penultimate": 2, "both_flags": 1,
                                         "last_only": 1, "penultimate_only": 1, "neither": 1})

    def test_selection_keeps_global_manifest_ids_and_covers_present_categories(self):
        raw, flags = validate_targets(self.samples)
        _, _, below, above = target_arrays(raw)
        chosen = select_examples([1, 3], raw, below, above, flags)
        self.assertEqual(list(chosen), [1, 3])
        self.assertIn("fwd_above_max", chosen[1])
        self.assertIn("yaw_above_max", chosen[1])
        self.assertIn("both_land_flags", chosen[3])
        self.assertNotIn("land_negative", {x for reasons in chosen.values() for x in reasons})

    def test_missing_nonfinite_and_float32_overflow_labels_fail_with_sample_id(self):
        for value in (None, "not-a-number", float("nan"), float("inf"), 1e100):
            records = copy.deepcopy(self.samples)
            records[2]["label"]["yaw"] = value
            with self.subTest(value=value), self.assertRaises(AuditValidationError) as raised:
                validate_targets(records)
            self.assertEqual(raised.exception.errors[0]["sample_id"], 2)
            self.assertEqual(raised.exception.errors[0]["field"], "label.yaw")
        records = copy.deepcopy(self.samples)
        del records[1]["label"]["fwd"]
        with self.assertRaises(AuditValidationError) as raised:
            validate_targets(records)
        self.assertEqual(raised.exception.errors[0]["sample_id"], 1)

    def test_invalid_flags_and_empty_instructions_are_reported_together(self):
        self.samples[0]["is_last_step"] = "false"
        self.samples[0]["instruction"] = " "
        with self.assertRaises(AuditValidationError) as raised:
            validate_targets(self.samples)
        self.assertEqual({error["field"] for error in raised.exception.errors}, {"is_last_step", "instruction"})


class AuditArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "run/supervision"
        self.output.parent.mkdir()
        self.data = self.root / "data.json"
        self.split = self.root / "split.json"
        self.config = self.output.parent / "run_config.json"
        self.samples = [sample("map/a", "0.png", (-1, 7, 0)), sample("map/b", "0.png"),
                        sample("map/a", "1.png", last=True), sample("map/b", "1.png", penultimate=True)]
        self.data.write_text(json.dumps(self.samples), encoding="utf-8")
        self.split.write_text(json.dumps({"version": 1, "split_by": "trajectory", "seed": 42,
                                          "train": ["map/a"], "val": ["map/b"],
                                          "trajectory_sha256": hashlib.sha256(b"map/a\nmap/b").hexdigest()}), encoding="utf-8")
        config = {"data_root": str(self.root / "images-not-created")}
        for name in ("train", "val"):
            _, config[f"{name}_split"] = select_trajectory_split(self.samples, self.split, name, data_json=self.data)
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def test_full_audit_artifacts_and_rng_preservation_without_image_access(self):
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        result = audit(self.data, self.split, self.output)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["sources"]["run_config_match"]["status"], "matched")
        self.assertEqual(result["splits"]["train"]["num_samples"], 2)
        self.assertEqual(result["splits"]["val"]["land"]["positive"], 1)
        rows = [json.loads(line) for line in (self.output / "samples.jsonl").read_text().splitlines()]
        self.assertEqual({row["sample_id"] for row in rows if row["split"] == "train"}, {0, 2})
        self.assertEqual({row["sample_id"] for row in rows if row["split"] == "val"}, {1, 3})
        first = rows[0]
        self.assertEqual(first["raw_label"], self.samples[0]["label"])
        self.assertEqual(first["instruction"], "  Fly forward  ")
        self.assertEqual(first["front_image_relative"], "map/a/frontcamera/0.png")
        self.assertEqual(first["clipping_axes"], ["fwd", "down"])
        self.assertIn("不是训练实际 batch", (self.output / "README.md").read_text())
        with (self.output / "axis_statistics.csv").open(encoding="utf-8-sig") as handle:
            stats = list(csv.DictReader(handle))
        self.assertEqual(stats[0]["count"], "2")
        self.assertEqual(stats[0]["raw_clipping_count"], "1")
        self.assertEqual(random.getstate(), python_rng)
        self.assertEqual(np.random.get_state()[0], numpy_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))

    def test_stale_source_hash_fails_and_writes_error_record(self):
        self.samples[0]["label"]["fwd"] = 0
        self.data.write_text(json.dumps(self.samples), encoding="utf-8")
        with self.assertRaises(AuditValidationError):
            audit(self.data, self.split, self.output)
        failure = json.loads((self.output / "audit_errors.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertTrue(any(error["field"].endswith("data_sha256") for error in failure["errors"]))
        self.assertFalse((self.output / "samples.csv").exists())

    def test_nonfinite_source_label_writes_explicit_error_without_silently_skipping(self):
        self.samples[0]["label"]["down"] = float("nan")
        self.data.write_text(json.dumps(self.samples), encoding="utf-8")
        with self.assertRaises(AuditValidationError):
            audit(self.data, self.split, self.output)
        failure = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["errors"][0]["field"], "label.down")
        self.assertEqual(failure["errors"][0]["sample_id"], 0)
        self.assertIn("nonfinite", failure["errors"][0]["error"])


if __name__ == "__main__":
    unittest.main()
