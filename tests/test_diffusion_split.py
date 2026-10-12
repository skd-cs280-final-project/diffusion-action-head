"""Trajectory isolation, stale-manifest detection and reproducible subsets."""

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diffusion_dataset import AeroVLADiffusionDataset  # noqa: E402
from diffusion_split import deterministic_subset, select_trajectory_split  # noqa: E402


def trajectory_digest(values):
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


class TrajectorySplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.split_path = self.root / "split.json"
        self.data_path = self.root / "data.json"
        self.samples = [
            {"traj_rel_dir": f"map/{trajectory}", "img_name": f"{frame:06d}.png"}
            for frame in range(8)
            for trajectory in ("a", "b", "c")
        ]
        self.manifest = {
            "version": 1,
            "seed": 42,
            "split_by": "trajectory",
            "trajectory_sha256": trajectory_digest(["map/a", "map/b", "map/c"]),
            "train": ["map/a", "map/c"],
            "val": ["map/b"],
        }
        self.write_files()

    def write_files(self):
        self.split_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        self.data_path.write_text(json.dumps(self.samples), encoding="utf-8")

    def dataset(self, split="train", **kwargs):
        return AeroVLADiffusionDataset(
            self.root, self.data_path, None, None,
            split_json=self.split_path, split=split, **kwargs,
        )

    def select(self, split="train"):
        self.write_files()
        return select_trajectory_split(self.samples, self.split_path, split, data_json=self.data_path)

    def test_dataset_selects_disjoint_complete_partitions_in_original_order(self):
        train, val = self.dataset(), self.dataset("val")
        self.assertEqual(train.samples, [sample for sample in self.samples if sample["traj_rel_dir"] != "map/b"])
        self.assertEqual(val.samples, [sample for sample in self.samples if sample["traj_rel_dir"] == "map/b"])
        self.assertEqual(len(train) + len(val), len(self.samples))
        self.assertFalse({x["traj_rel_dir"] for x in train.samples} & {x["traj_rel_dir"] for x in val.samples})
        self.assertEqual(train.split_info["train_samples"], 16)
        self.assertEqual(val.split_info["num_trajectories"], 1)
        self.assertEqual(val.split_info["total_trajectories"], 3)
        self.assertEqual(train.split_info["split_sha256"], hashlib.sha256(self.split_path.read_bytes()).hexdigest())
        self.assertEqual(train.split_info["data_sha256"], hashlib.sha256(self.data_path.read_bytes()).hexdigest())
        self.assertEqual(train.split_info["data_json"], str(self.data_path.resolve()))
        self.assertEqual(train.split_info["train_trajectory_sha256"], trajectory_digest(["map/a", "map/c"]))
        json.dumps(train.split_info)  # Metadata can be recorded in checkpoints.

    def test_partition_fingerprints_change_when_assignment_changes(self):
        _, original = self.select()
        self.manifest["train"], self.manifest["val"] = self.manifest["val"], self.manifest["train"]
        _, changed = self.select()
        self.assertEqual(original["trajectory_sha256"], changed["trajectory_sha256"])
        for key in ("split_sha256", "train_trajectory_sha256", "val_trajectory_sha256", "selected_indices_sha256"):
            self.assertNotEqual(original[key], changed[key])

    def test_bad_manifest_schema_rejected(self):
        original = copy.deepcopy(self.manifest)
        for field, values in (
            ("version", (None, True, 1.0, 2, "1")),
            ("split_by", (None, "frame")),
            ("train", (None, [], "map/a")),
            ("val", (None, [], [None])),
            ("trajectory_sha256", (None, "abc", "g" * 64, "0" * 64)),
        ):
            for value in values:
                self.manifest = dict(original, **{field: value})
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.select()
        self.manifest = []
        with self.assertRaisesRegex(ValueError, "object"):
            self.select()

    def test_duplicate_and_overlapping_trajectory_ids_rejected(self):
        self.manifest["train"].append("map/a")
        with self.assertRaisesRegex(ValueError, "duplicate trajectory"):
            self.select()
        self.manifest["train"].pop()
        self.manifest["val"].append("map/a")
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.select()

    def test_missing_or_extra_trajectories_rejected_after_valid_digest(self):
        self.manifest["train"] = ["map/a", "map/missing"]
        self.manifest["trajectory_sha256"] = trajectory_digest(self.manifest["train"] + self.manifest["val"])
        with self.assertRaisesRegex(ValueError, "1 missing from split.*1 absent from dataset"):
            self.select()

    def test_duplicate_frame_rejected_even_when_not_in_selected_partition(self):
        self.samples.append(dict(self.samples[1]))  # Duplicate belongs to val.
        with self.assertRaisesRegex(ValueError, "Duplicate frame"):
            self.select("train")

    def test_invalid_records_and_path_aliases_rejected(self):
        original = self.samples[0]
        for sample in (None, {}, dict(original, traj_rel_dir="map/../map/a"),
                       dict(original, traj_rel_dir="/map/a"), dict(original, traj_rel_dir="map//a"),
                       dict(original, img_name="other/000.png")):
            self.samples[0] = sample
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                self.select()
        self.samples = []
        with self.assertRaisesRegex(ValueError, "nonempty list"):
            self.select()

    def test_partition_name_and_manifest_must_be_paired(self):
        for name in (None, "test", "all", 0):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "train.*val"):
                self.dataset(name)
        with self.assertRaisesRegex(ValueError, "split requires split_json"):
            AeroVLADiffusionDataset(self.root, self.data_path, None, None, split="val")
        legacy = AeroVLADiffusionDataset(self.root, self.data_path, None, None)
        self.assertEqual(len(legacy), len(self.samples))
        self.assertIsNone(legacy.split_info["split"])

    def test_subset_reproducibility_rng_isolation_and_metadata(self):
        dataset = self.dataset()
        original_metadata = copy.deepcopy(dataset.split_info)
        state = random.getstate()
        subset = deterministic_subset(dataset, 7, 42)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(subset.indices, deterministic_subset(dataset, 7, 42).indices)
        self.assertNotEqual(subset.indices, deterministic_subset(dataset, 7, 43).indices)
        self.assertEqual(subset.indices, sorted(subset.indices))
        self.assertNotEqual(subset.indices, list(range(7)))
        self.assertEqual(dataset.split_info, original_metadata)
        self.assertEqual(subset.split_info["num_samples"], 7)
        self.assertEqual(subset.split_info["split_sha256"], dataset.split_info["split_sha256"])
        expected = {dataset.samples[index]["traj_rel_dir"] for index in subset.indices}
        self.assertEqual(subset.split_info["num_trajectories"], len(expected))
        self.assertEqual(subset.split_info["subset"]["source_num_samples"], len(dataset))
        digest = hashlib.sha256("\n".join(map(str, subset.indices)).encode()).hexdigest()
        self.assertEqual(subset.split_info["subset"]["indices_sha256"], digest)
        json.dumps(subset.split_info)

    def test_subset_limits_and_nested_subsets(self):
        dataset = self.dataset()
        for limit in (None, len(dataset), len(dataset) + 1):
            self.assertIs(deterministic_subset(dataset, limit, 42), dataset)
        for limit in (0, -1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "positive integer"):
                deterministic_subset(dataset, limit, 42)
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            deterministic_subset(dataset, 2, True)
        subset = deterministic_subset(dataset, 7, 42)
        nested = deterministic_subset(subset, 3, 43)
        expected = {dataset.samples[subset.indices[index]]["traj_rel_dir"] for index in nested.indices}
        self.assertEqual(nested.split_info["num_trajectories"], len(expected))


if __name__ == "__main__":
    unittest.main()
