"""Validated trajectory splits and reproducible frame subsets for diffusion.

Split version 1 fingerprints the complete trajectory set with SHA256 over
``"\n".join(sorted(trajectory_ids))`` (UTF-8, no trailing newline). Both the
training and evaluation entrypoints use this module to avoid split drift.
"""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import random
import re

from torch.utils.data import Subset


def _sha256_lines(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _relative_identifier(value, field):
    # Alias paths (e.g. a/../b) must not let the same physical trajectory cross
    # a train/val boundary under different JSON identifiers.
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a nonempty canonical relative path; got {value!r}.")
    return value


def select_trajectory_split(samples, split_json, split, *, data_json=None):
    """Validate the full manifest and dataset, then return records and metadata.

    Validation always covers both partitions and every original frame, before
    selection or any smoke-test subsampling. Unknown or omitted trajectories,
    overlapping partitions, duplicate frames and stale fingerprints all fail.
    """
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val' when split_json is provided.")
    if not isinstance(samples, list) or not samples:
        raise ValueError("A trajectory split requires a nonempty list of frame-label records.")
    manifest_path = Path(split_json).expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("The split JSON must contain an object.")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise ValueError("Only split manifest version 1 is supported.")
    if manifest.get("split_by") != "trajectory":
        raise ValueError("The split manifest must use split_by='trajectory'.")

    partitions = {}
    for name in ("train", "val"):
        identifiers = manifest.get(name)
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"Split '{name}' must be a nonempty list of trajectory IDs.")
        for identifier in identifiers:
            _relative_identifier(identifier, f"{name} trajectory ID")
        partitions[name] = set(identifiers)
        if len(partitions[name]) != len(identifiers):
            raise ValueError(f"Split '{name}' contains duplicate trajectory IDs.")
    overlap = partitions["train"] & partitions["val"]
    if overlap:
        raise ValueError(f"Train/val trajectory overlap: {sorted(overlap)[:3]}.")
    all_manifest_trajectories = partitions["train"] | partitions["val"]
    trajectory_sha256 = _sha256_lines(sorted(all_manifest_trajectories))
    recorded_sha256 = manifest.get("trajectory_sha256")
    if not isinstance(recorded_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", recorded_sha256):
        raise ValueError("trajectory_sha256 must contain a 64-character SHA256 digest.")
    if recorded_sha256.lower() != trajectory_sha256:
        raise ValueError("Split trajectory_sha256 does not match its train/val trajectory IDs.")

    frames = set()
    dataset_trajectories = set()
    frame_counts = {"train": 0, "val": 0}
    selected_indices = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"Dataset sample at index {index} must be an object.")
        trajectory = _relative_identifier(sample.get("traj_rel_dir"), f"sample {index} traj_rel_dir")
        image_name = _relative_identifier(sample.get("img_name"), f"sample {index} img_name")
        if "/" in image_name:
            raise ValueError(f"sample {index} img_name must be a filename, without directory components.")
        frame = (trajectory, image_name)
        if frame in frames:
            raise ValueError(f"Duplicate frame at dataset index {index}: {trajectory}/{image_name}.")
        frames.add(frame)
        dataset_trajectories.add(trajectory)
        for name in ("train", "val"):
            if trajectory in partitions[name]:
                frame_counts[name] += 1
        if trajectory in partitions[split]:
            selected_indices.append(index)

    missing = dataset_trajectories - all_manifest_trajectories
    extra = all_manifest_trajectories - dataset_trajectories
    if missing or extra:
        raise ValueError(
            "Split must exactly cover the dataset trajectories: "
            f"{len(missing)} missing from split {sorted(missing)[:3]}, "
            f"{len(extra)} absent from dataset {sorted(extra)[:3]}."
        )
    info = {
        "split": split,
        "split_json": str(manifest_path),
        "split_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "version": 1,
        "split_by": "trajectory",
        "seed": manifest.get("seed"),
        "trajectory_sha256": trajectory_sha256,
        "train_trajectory_sha256": _sha256_lines(sorted(partitions["train"])),
        "val_trajectory_sha256": _sha256_lines(sorted(partitions["val"])),
        "total_samples": len(samples),
        "total_trajectories": len(dataset_trajectories),
        "num_samples": len(selected_indices),
        "num_trajectories": len(partitions[split]),
        "train_samples": frame_counts["train"],
        "val_samples": frame_counts["val"],
        "train_trajectories": len(partitions["train"]),
        "val_trajectories": len(partitions["val"]),
        "selected_indices_sha256": _sha256_lines(selected_indices),
    }
    if data_json is not None:
        data_path = Path(data_json).expanduser().resolve()
        digest = hashlib.sha256()
        with data_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        info.update({"data_json": str(data_path), "data_sha256": digest.hexdigest()})
    return [samples[index] for index in selected_indices], info


def _sample_record(dataset, index):
    """Inspect metadata without loading image pixels, including nested subsets."""
    while isinstance(dataset, Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return dataset.samples[index] if hasattr(dataset, "samples") else None


def deterministic_subset(dataset, max_samples, seed):
    """Select a seeded random frame subset, preserving its original order.

    ``None`` means the full dataset; zero and negative limits are errors. A
    private Python RNG keeps selection independent of model/data-loader RNGs.
    The returned dataset exposes ``split_info`` when sampling is requested.
    """
    if max_samples is None:
        return dataset
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError("max_samples must be a positive integer or None.")
    if type(seed) is not int:
        raise ValueError("Subset seed must be an integer.")
    if max_samples >= len(dataset):
        return dataset
    indices = sorted(random.Random(seed).sample(range(len(dataset)), max_samples))
    subset = Subset(dataset, indices)
    info = dict(getattr(dataset, "split_info", {}))
    records = [_sample_record(dataset, index) for index in indices]
    if all(isinstance(record, Mapping) and "traj_rel_dir" in record for record in records):
        info["num_trajectories"] = len({record["traj_rel_dir"] for record in records})
    else:
        info.pop("num_trajectories", None)
    info.update({
        "num_samples": len(indices),
        "subset": {
            "seed": seed,
            "max_samples": max_samples,
            "source_num_samples": len(dataset),
            "selection": "seeded_random_frames_in_original_order",
            "indices_sha256": _sha256_lines(indices),
        },
    })
    subset.split_info = info
    return subset
