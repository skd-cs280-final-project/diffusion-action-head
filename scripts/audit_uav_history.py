#!/usr/bin/env python3
"""Audit the source-audit sample through the actual causal history dataset path.

Run from any directory with the AeroVLA Python environment. This reads no image
pixels, runs no model, and creates only feature_audit.json/history_examples.png.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import diffusion_dataset as dataset_module
from src.uav_history import HISTORY_SCHEMA, STATE_FEATURE_NAMES, STATE_VALUE_DIM, observation_identity

PLOT_SELECTION = (
    "Choose full histories on three distinct maps: largest observed horizontal path turn "
    "(requiring consecutive segments >=0.25m), largest net climb, then largest net descent. "
    "Selection uses only each past/current window and is illustrative, not representative sampling."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_history(dataset, sample):
    """Capture precisely the logs the production loader reads, all at t <= now."""
    accessed = []
    original_reader = dataset_module._read_history_log

    def capture(path):
        raw = original_reader(path)
        record = dict(raw)
        frame = int(Path(path).stem)
        record.setdefault("frame_id", frame)
        accessed.append((frame, record))
        return raw

    with patch.object(dataset_module, "_read_history_log", capture):
        encoded, mask = dataset._get_state_history(sample)
    frame = int(Path(sample["img_name"]).stem)
    assert accessed and all(past_frame <= frame for past_frame, _ in accessed)
    selected, seen = [], set()
    for past_frame, record in accessed:  # Actual loader reads current to oldest.
        identity = observation_identity(record)
        if identity not in seen:
            seen.add(identity)
            selected.append((past_frame, record))
        if len(selected) == dataset.history_length:
            break
    selected.reverse()
    assert selected[-1][0] == frame
    assert len(selected) == int(mask.sum())
    states = [record["sensors"]["state"] for _, record in selected]
    positions = np.asarray([state["position"] for state in states], dtype=np.float64)
    # Independent SciPy conversion verifies the implementation's quaternion math.
    angles = Rotation.from_quat([state["orientation"] for state in states]).as_euler("xyz")
    yaw = angles[-1, 2]
    world_relative = positions - positions[-1]
    local = world_relative @ np.asarray([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1],
    ])
    relative_yaw = angles[:, 2] - yaw
    preclip = np.column_stack((
        local / HISTORY_SCHEMA["position_scale_m"],
        np.sin(angles[:, 0]), np.cos(angles[:, 0]),
        np.sin(angles[:, 1]), np.cos(angles[:, 1]),
        np.sin(relative_yaw), np.cos(relative_yaw),
    ))
    values = encoded.numpy()
    valid_values = values[mask.numpy()]
    error = float(np.max(np.abs(valid_values[:, :9] - np.clip(preclip, -1, 1))))
    assert error < 1e-6, f"Independent geometry mismatch: {error}"
    assert np.isin(valid_values[:, STATE_VALUE_DIM:], [0, 1]).all()
    clipped = np.abs(preclip) > 1.0
    yaw_deltas = np.arctan2(np.sin(np.diff(angles[:, 2])), np.cos(np.diff(angles[:, 2])))
    return {
        "trajectory": sample["traj_rel_dir"], "img_name": sample["img_name"],
        "frame": frame, "split": dataset.split_info["split"],
        "loaded_frames": [past_frame for past_frame, _ in accessed],
        "selected_frames": [past_frame for past_frame, _ in selected],
        "valid_slots": len(selected), "history_mask": mask.tolist(),
        "nonfinite_scalar_count": int((~np.isfinite(values)).sum()),
        "velocity_valid_slots_xyz": (valid_values[:, STATE_VALUE_DIM + 9:STATE_VALUE_DIM + 12] == 1).sum(axis=0).tolist(),
        "time_valid_slots": int((valid_values[:, STATE_VALUE_DIM + 12] == 1).sum()),
        "first9_clip_counts": clipped.sum(axis=0).tolist(),
        "first9_coordinate_count": int(preclip.size),
        "independent_geometry_max_abs_error": error,
        "local_xyz_m": local.tolist(), "relative_yaw_rad": relative_yaw.tolist(),
        "history_values_and_validity": values.tolist(),
        "absolute_yaw_change_degrees": float(np.degrees(np.abs(yaw_deltas).sum())),
        "vertical_range_m": float(np.ptp(local[:, 2])),
    }


def select_examples(results, history_length):
    candidates = [item for item in results if item["valid_slots"] == history_length]

    def path_turn(item):
        delta = np.diff(np.asarray(item["local_xyz_m"])[:, :2], axis=0)
        angles = []
        for a, b in zip(delta, delta[1:]):
            if min(np.linalg.norm(a), np.linalg.norm(b)) >= 0.25:
                angles.append(abs(math.atan2(a[0] * b[1] - a[1] * b[0], float(a @ b))))
        return sum(angles)

    scores = (("Turning", path_turn),
              ("Climbing", lambda item: item["local_xyz_m"][0][2]),
              ("Descending", lambda item: -item["local_xyz_m"][0][2]))
    examples, maps = [], set()
    for reason, score in scores:
        remaining = [item for item in candidates if item["trajectory"].split("/")[0] not in maps]
        if not remaining:
            raise ValueError("Need three maps with full history for the plot")
        item = dict(max(remaining, key=score))
        item["illustration_reason"] = reason
        examples.append(item)
        maps.add(item["trajectory"].split("/")[0])
    return examples


def draw_examples(examples, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aerovla-history-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, len(examples), figsize=(14, 7), squeeze=False, constrained_layout=True)
    fig.suptitle("Actual UAV observation history (K=4): only past and current states", fontsize=16)
    for column, item in enumerate(examples):
        xyz = np.asarray(item["local_xyz_m"])
        yaw = np.asarray(item["relative_yaw_rad"])
        count = len(xyz)
        slots = np.arange(-count + 1, 1)
        ax = axes[0, column]
        ax.plot(xyz[:, 0], xyz[:, 1], color="#526778", linewidth=1.8, zorder=1)
        ax.scatter(xyz[:, 0], xyz[:, 1], c=slots, cmap="viridis", s=65, zorder=3)
        scale = max(np.ptp(xyz[:, :2], axis=0).max(), 1) * 0.09
        ax.quiver(xyz[:, 0], xyz[:, 1], np.cos(yaw) * scale, np.sin(yaw) * scale,
                  angles="xy", scale_units="xy", scale=1, color="#526778", width=0.009)
        for slot, point in zip(slots, xyz):
            ax.annotate(f"t{slot:+d}" if slot else "current", point[:2], xytext=(5, 7),
                        textcoords="offset points", fontsize=9)
        ax.scatter([0], [0], marker="*", s=180, color="#c64635", zorder=5)
        ax.axhline(0, color="#d8dee3", linewidth=0.7)
        ax.axvline(0, color="#d8dee3", linewidth=0.7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Local x / current forward (m)")
        ax.set_ylabel("Local y / current right (m)")
        ax.grid(alpha=0.16)
        map_name, trajectory_id = item["trajectory"].split("/", 1)
        ax.set_title(f"{item['illustration_reason']}: {map_name} / {item['split']}\n{trajectory_id[:8]} | current frame {item['frame']}", fontsize=11)
        bottom = axes[1, column]
        bottom.plot(slots, xyz[:, 2], marker="o", color="#267b8c", linewidth=2)
        bottom.scatter([0], [0], marker="*", s=150, color="#c64635", zorder=5)
        bottom.axhline(0, color="#d8dee3", linewidth=0.7)
        bottom.set_xticks(slots, [f"{slot}\nframe {frame}" for slot, frame in zip(slots, item["selected_frames"])])
        bottom.set_xlabel("Observation slot relative to current (not seconds)")
        bottom.set_ylabel("Relative z (m; NED down positive)")
        bottom.grid(alpha=0.16)
        bottom.set_title(f"Yaw change {item['absolute_yaw_change_degrees']:.1f} degrees; z range {item['vertical_range_m']:.1f} m", fontsize=11)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset_raw")
    parser.add_argument("--data-json", type=Path, default=ROOT / "data/aerovla_train_dataset.json")
    parser.add_argument("--split-json", type=Path, default=ROOT / "data/continuous_split_seed42.json")
    parser.add_argument("--source-audit", type=Path, default=ROOT / "docs/reports/history_state_20261012/source_audit.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/reports/history_state_20261012")
    parser.add_argument("--history-length", type=int, default=4)
    args = parser.parse_args()
    source = json.loads(args.source_audit.read_text())
    assert sha256(args.data_json) == source["data_json_sha256"], "Source audit belongs to different data JSON"
    targets = {(item["trajectory"], item["img_name"]): item for item in source["samples"]}
    assert len(targets) == source["sample_size"]
    results, failures, split_info = [], [], {}
    for split in ("train", "val"):
        dataset = dataset_module.AeroVLADiffusionDataset(
            args.data_root, args.data_json, tokenizer=None, processor=None,
            split_json=args.split_json, split=split, history_length=args.history_length,
        )
        split_info[split] = dataset.split_info
        for sample in dataset.samples:
            key = (sample["traj_rel_dir"], sample["img_name"])
            if key not in targets:
                continue
            try:
                results.append(inspect_history(dataset, sample))
            except Exception as exc:
                failures.append({"trajectory": key[0], "img_name": key[1], "split": split,
                                 "error": f"{type(exc).__name__}: {exc}"})
        del dataset
        gc.collect()
    assert len(results) + len(failures) == len(targets), "Some source audit samples were not in the split datasets"
    clips = np.sum([item["first9_clip_counts"] for item in results], axis=0)
    coordinates = sum(item["first9_coordinate_count"] for item in results)
    examples = select_examples(results, args.history_length)
    summary = {
        "samples_requested": len(targets), "samples_succeeded": len(results), "samples_failed": len(failures),
        "split_sample_counts": dict(Counter(item["split"] for item in results)),
        "valid_slot_count_distribution": dict(sorted(Counter(item["valid_slots"] for item in results).items())),
        "total_valid_slots": sum(item["valid_slots"] for item in results),
        "nonfinite_scalar_count": sum(item["nonfinite_scalar_count"] for item in results),
        "velocity_valid_slots_xyz": np.sum([item["velocity_valid_slots_xyz"] for item in results], axis=0).tolist(),
        "time_valid_slots": sum(item["time_valid_slots"] for item in results),
        "samples_with_any_velocity": sum(any(item["velocity_valid_slots_xyz"]) for item in results),
        "samples_with_any_time": sum(item["time_valid_slots"] > 0 for item in results),
        "first9_clip_count": int(clips.sum()), "first9_coordinate_count": coordinates,
        "first9_clip_fraction": float(clips.sum() / coordinates),
        "first9_clip_by_feature": dict(zip(STATE_FEATURE_NAMES[:9], clips.tolist())),
        "max_independent_geometry_abs_error": max(item["independent_geometry_max_abs_error"] for item in results),
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Fixed source-audit sample, not full dataset; feature encoding only, no model or performance claim.",
        "source_audit": str(args.source_audit.resolve()), "source_audit_sha256": sha256(args.source_audit),
        "source_seed": source["seed"], "history_length": args.history_length, "history_schema": HISTORY_SCHEMA,
        "data_json_sha256": sha256(args.data_json), "split_info": split_info,
        "method": "Call AeroVLADiffusionDataset._get_state_history in the sample's validated train/val split. Capture exactly the accessed past/current log files. No image pixels, future logs, merged trajectories or action labels used for features or plots. Independent SciPy xyz Euler conversion checks the first nine coordinates.",
        "clipping_definition": "Fraction of valid preclip coordinates with absolute value > 1, before clipping; natural sin/cos endpoints at +/-1 are not counted as clipping.",
        "plot_selection": PLOT_SELECTION,
        "plot_examples": [{key: item[key] for key in ("trajectory", "frame", "split", "selected_frames", "absolute_yaw_change_degrees", "vertical_range_m", "illustration_reason")} for item in examples],
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                        (Path(__file__), ROOT / "src/diffusion_dataset.py", ROOT / "src/uav_history.py")},
        "summary": summary, "failures": failures, "samples": results,
        "reproduce": f"/data/xiyang/.conda/envs/aero_vla/bin/python scripts/audit_uav_history.py --history-length {args.history_length}",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_examples(examples, args.output_dir / "history_examples.png")
    (args.output_dir / "feature_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if failures or summary["nonfinite_scalar_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
