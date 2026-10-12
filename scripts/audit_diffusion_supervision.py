#!/usr/bin/env python3
"""Audit frozen manifest targets without models, tokenization or image decoding.

The selected examples are deterministic inspection examples, never a trace of
training batches. No model predictions or CUDA operations are performed.
"""

import argparse
from collections import defaultdict
from collections.abc import Mapping
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from diffusion_dataset import ACTION_MAX, ACTION_MIN, ACTION_ORDER, normalize_actions  # noqa: E402
from diffusion_split import select_trajectory_split  # noqa: E402

DEFAULT_RUN = "checkpoints/aero_vla_diffusion_full_20261010_200045"
LAND_RULE = "land = int(is_last_step or is_penultimate); normalized_land = 2 * land - 1"


class AuditValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Supervision audit failed with {len(errors)} validation error(s).")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_targets(samples):
    """Validate every label/flag; preserve global manifest indices in failures."""
    if not isinstance(samples, list) or not samples:
        raise AuditValidationError([{"field": "dataset", "error": "expected nonempty list"}])
    raw = np.zeros((len(samples), 4), dtype=np.float64)
    flags = np.zeros((len(samples), 2), dtype=bool)
    errors = []
    for sample_id, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append({"sample_id": sample_id, "field": "sample", "error": "expected object"})
            continue

        def error(field, reason):
            errors.append({"sample_id": sample_id, "traj_rel_dir": sample.get("traj_rel_dir"),
                           "img_name": sample.get("img_name"), "field": field, "error": reason})

        instruction = sample.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            error("instruction", "missing or empty string")
        label = sample.get("label")
        if not isinstance(label, Mapping):
            error("label", "missing or not an object")
        else:
            for axis, name in enumerate(ACTION_ORDER[:3]):
                try:
                    value = float(label[name])
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    error(f"label.{name}", f"missing or nonnumeric: {exc}")
                    continue
                if not math.isfinite(value):
                    error(f"label.{name}", f"nonfinite value: {value!r}")
                elif abs(value) > np.finfo(np.float32).max:
                    error(f"label.{name}", "not finite after training float32 conversion")
                else:
                    raw[sample_id, axis] = value
        for axis, name in enumerate(("is_last_step", "is_penultimate")):
            value = sample.get(name)
            if not isinstance(value, (bool, int)) or value not in (0, 1):
                error(name, "missing or not a boolean/0/1")
            else:
                flags[sample_id, axis] = value
        raw[sample_id, 3] = float(flags[sample_id].any())
    if errors:
        raise AuditValidationError(errors)
    return raw, flags


def target_arrays(raw):
    """Use the actual training normalizer on CPU, retaining raw float64 labels."""
    physical = torch.from_numpy(raw.astype(np.float32))
    lower, upper = physical.new_tensor(ACTION_MIN), physical.new_tensor(ACTION_MAX)
    clipped = torch.minimum(torch.maximum(physical, lower), upper).numpy()
    normalized = normalize_actions(physical).numpy()
    # Count out-of-range source labels against the declared physical limits.
    below = raw < np.asarray(ACTION_MIN, dtype=np.float64)
    above = raw > np.asarray(ACTION_MAX, dtype=np.float64)
    return clipped, normalized, below, above


def axis_statistics(values):
    return {
        name: {"count": len(values), "min": float(values[:, axis].min()),
               "max": float(values[:, axis].max()),
               "mean": float(values[:, axis].mean(dtype=np.float64)),
               "std_population": float(values[:, axis].std(dtype=np.float64, ddof=0))}
        for axis, name in enumerate(ACTION_ORDER)
    }


def group_statistics(indices, raw, clipped, normalized, below, above, flags):
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    land = raw[indices, 3].astype(bool)
    last, penultimate = flags[indices, 0], flags[indices, 1]
    return {
        "num_samples": n,
        "raw_4d": axis_statistics(raw[indices]),
        "clipped_4d_float32": axis_statistics(clipped[indices]),
        "normalized_4d_float32": axis_statistics(normalized[indices]),
        "clipping": {
            name: {"below_min": int(below[indices, axis].sum()),
                   "above_max": int(above[indices, axis].sum()),
                   "count": int((below[indices, axis] | above[indices, axis]).sum()),
                   "ratio": float((below[indices, axis] | above[indices, axis]).mean())}
            for axis, name in enumerate(ACTION_ORDER)
        },
        "any_axis_clipped": int((below[indices] | above[indices]).any(axis=1).sum()),
        "land": {"positive": int(land.sum()), "negative": int((~land).sum()),
                 "positive_ratio": float(land.mean()), "last": int(last.sum()),
                 "penultimate": int(penultimate.sum()), "both_flags": int((last & penultimate).sum()),
                 "last_only": int((last & ~penultimate).sum()),
                 "penultimate_only": int((~last & penultimate).sum()), "neither": int((~land).sum())},
    }


def select_examples(indices, raw, below, above, flags):
    """First example of every available audit category, with no RNG use."""
    chosen = defaultdict(list)
    seen = set()
    for sample_id in indices:
        categories = ["first_in_map", "land_positive" if raw[sample_id, 3] else "land_negative"]
        if flags[sample_id, 0]:
            categories.append("last_step")
        if flags[sample_id, 1]:
            categories.append("penultimate_step")
        if flags[sample_id].all():
            categories.append("both_land_flags")
        if (below[sample_id] | above[sample_id]).any():
            categories.append("any_clipping")
        for axis, name in enumerate(ACTION_ORDER[:3]):
            if below[sample_id, axis]:
                categories.append(f"{name}_below_min")
            if above[sample_id, axis]:
                categories.append(f"{name}_above_max")
        for category in categories:
            if category not in seen:
                chosen[sample_id].append(category)
                seen.add(category)
    return dict(sorted(chosen.items()))


def verify_run_sources(run_config, split_infos):
    errors = []
    keys = ("data_sha256", "split_sha256", "trajectory_sha256", "selected_indices_sha256",
            "num_samples", "num_trajectories", "train_trajectory_sha256", "val_trajectory_sha256")
    for split, info in split_infos.items():
        expected = run_config.get(f"{split}_split", {})
        for key in keys:
            if expected.get(key) != info.get(key):
                errors.append({"field": f"run_config.{split}_split.{key}",
                               "expected": expected.get(key), "actual": info.get(key),
                               "error": "run configuration and audit sources do not match"})
    if errors:
        raise AuditValidationError(errors)
    return {"status": "matched", "checked_fields_per_split": list(keys)}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def markdown_summary(summary, map_rows):
    lines = ["# 扩散头监督标签审计", "", f"生成时间（UTC）：{summary['created_at_utc']}。", "",
             "这是针对当前 run 数据来源的全量静态标签审计；固定审计样本不是训练实际 batch 顺序，文件不包含模型预测。",
             "不加载模型、不分词、不解码图片、不使用 GPU，也不访问训练进程的随机数状态。", "",
             "## 标签规则", "", "动作顺序为 `fwd, down, yaw, land`，每帧一个 4D 动作。",
             "`fwd` 裁剪至 `[0, 5]`，`down` 至 `[-5, 5]`，`yaw` 至 `[-1.1, 1.1]`；",
             "`land = int(is_last_step or is_penultimate)`。归一化使用训练的 float32 函数映射至 `[-1, 1]`。",
             "原始范围/均值/总体标准差来自原始数值（float64）；裁剪计数以原始标签超出声明上下界判定。",
             "LAND 的 raw 统计指由两个布尔字段导出的 0/1，不表示 JSON 中存在独立 LAND 数值。", "",
             "## 来源与有效性", "", "全量标签、指令及 LAND 字段检查通过；没有跳过非法记录。",
             "数据、split、轨迹集合和选中索引 SHA256 与 `run_config.json` 的 train/val 记录完全匹配。", "",
             f"- 数据 SHA256：`{summary['sources']['data_sha256']}`",
             f"- Split SHA256：`{summary['sources']['split_sha256']}`",
             f"- run_config SHA256：`{summary['sources']['run_config_sha256']}`", "",
             "## Train/val 全量统计", "",
             "| split | 帧数 | 轨迹数 | LAND+ | LAND− | last | penultimate | 两者同时 | 任一轴裁剪 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, stats in summary["splits"].items():
        land = stats["land"]
        lines.append(f"| {split} | {stats['num_samples']} | {stats['num_trajectories']} | {land['positive']} | "
                     f"{land['negative']} | {land['last']} | {land['penultimate']} | {land['both_flags']} | "
                     f"{stats['any_axis_clipped']} |")
    lines += ["", "| split | 轴 | raw min | raw max | raw mean | raw std | 下界外 | 上界外 | 裁剪比例 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, stats in summary["splits"].items():
        for axis in ACTION_ORDER:
            raw, clip = stats["raw_4d"][axis], stats["clipping"][axis]
            lines.append(f"| {split} | {axis} | {raw['min']:.8g} | {raw['max']:.8g} | {raw['mean']:.8g} | "
                         f"{raw['std_population']:.8g} | {clip['below_min']} | {clip['above_max']} | {clip['ratio']:.4%} |")
    lines += ["", "## 地图分布", "", "| split | 地图 | 帧数 | 轨迹数 | LAND+ | 任一轴裁剪 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in map_rows:
        lines.append(f"| {row['split']} | {row['map']} | {row['num_samples']} | {row['num_trajectories']} | "
                     f"{row['land_positive']} | {row['any_axis_clipped']} |")
    lines += ["", "## 检查文件", "",
              f"- [samples.csv](samples.csv) / [samples.jsonl](samples.jsonl)：{summary['audit_examples']['num_samples']} 个固定审计样本，含原始全清单 0-based sample_id、原始 label、物理/裁剪/归一化动作、LAND 规则、指令、前/下视图路径。",
              "- [summary.json](summary.json)：完整统计、来源哈希、归一化规则和逐地图样本选取类别。",
              "- [map_distribution.csv](map_distribution.csv)：各 split/map 帧数、轨迹数、LAND 与裁剪分布。",
              "- [trajectory_distribution.csv](trajectory_distribution.csv)：每条轨迹帧数、LAND 计数及原始全清单首/末索引。",
              "- [axis_statistics.csv](axis_statistics.csv)：raw/clipped/normalized 三组逐轴统计。",
              "- [source_manifest.json](source_manifest.json)：源文件位置、哈希以及 run_config 匹配结果。", "",
              "固定样本按每个 split/map 中原始清单顺序选取各类别首例：LAND 正负、last、penultimate、同时标记、任一裁剪、每个连续轴上下越界。",
              "一个样本可覆盖多个类别；只选取确实存在的类别，不伪造缺失类别。图片只记录路径，不复制或嵌入。",
              f"完整 {sum(stats['num_samples'] for stats in summary['splits'].values())} 帧监督标签仍保留于原始 JSON；本目录的 CSV/JSONL 是便于人工抽查的固定子集。", ""]
    return "\n".join(lines)


def audit(data_json, split_json, output_dir, run_config_path=None):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_json, split_json = Path(data_json).resolve(), Path(split_json).resolve()
    run_config_path = Path(run_config_path).resolve() if run_config_path else output_dir.parent / "run_config.json"
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        samples = json.loads(data_json.read_text(encoding="utf-8"))
        raw, flags = validate_targets(samples)
        split_infos = {}
        for split in ("train", "val"):
            _, split_infos[split] = select_trajectory_split(samples, split_json, split, data_json=data_json)
        source_check = verify_run_sources(run_config, split_infos)
        data_root = Path(run_config["data_root"]).expanduser()
        if not data_root.is_absolute():
            data_root = PROJECT_ROOT / data_root
        data_root = data_root.resolve()
        manifest = json.loads(split_json.read_text(encoding="utf-8"))
        trajectory_split = {trajectory: split for split in ("train", "val") for trajectory in manifest[split]}
        groups, split_indices, trajectory_indices = defaultdict(list), defaultdict(list), defaultdict(list)
        for sample_id, sample in enumerate(samples):
            trajectory = sample["traj_rel_dir"]
            split = trajectory_split[trajectory]
            map_name = trajectory.split("/")[0]
            groups[split, map_name].append(sample_id)
            split_indices[split].append(sample_id)
            trajectory_indices[split, map_name, trajectory].append(sample_id)
        clipped, normalized, below, above = target_arrays(raw)
        sources = {"data_json": str(data_json), "data_sha256": split_infos["train"]["data_sha256"],
                   "split_json": str(split_json), "split_sha256": split_infos["train"]["split_sha256"],
                   "run_config": str(run_config_path), "run_config_sha256": sha256_file(run_config_path),
                   "audit_script": str(Path(__file__).resolve()), "audit_script_sha256": sha256_file(__file__),
                   "normalizer_sha256": sha256_file(PROJECT_ROOT / "src/diffusion_dataset.py"),
                   "split_validator_sha256": sha256_file(PROJECT_ROOT / "src/diffusion_split.py"),
                   "action_stats_sha256": sha256_file(PROJECT_ROOT / "src/aerovla_dataset.py"),
                   "data_root": str(data_root), "run_config_match": source_check,
                   "validated_splits": split_infos}
        summary = {"status": "passed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "scope": "full_manifest_static_targets; fixed_examples_not_training_batch_order; no_predictions",
                   "sources": sources, "num_invalid_samples": 0,
                   "target_contract": {"action_order": ACTION_ORDER, "action_horizon": 1,
                                       "minimum": ACTION_MIN, "maximum": ACTION_MAX, "land_rule": LAND_RULE,
                                       "normalization": "training float32 normalize_actions on CPU",
                                       "std_definition": "population ddof=0",
                                       "clipping_count_definition": "raw float64 source value outside declared bounds"},
                   "splits": {}, "audit_examples": {"selection": "first_global_manifest_index_per_split_map_category",
                                                       "num_samples": 0, "categories_per_map": {}}}
        stats_rows, map_rows, trajectory_rows, examples = [], [], [], []
        for split in ("train", "val"):
            stats = group_statistics(split_indices[split], raw, clipped, normalized, below, above, flags)
            stats["num_trajectories"] = split_infos[split]["num_trajectories"]
            summary["splits"][split] = stats
            for representation in ("raw_4d", "clipped_4d_float32", "normalized_4d_float32"):
                for axis in ACTION_ORDER:
                    stats_rows.append({"split": split, "representation": representation,
                                       "axis": axis, **stats[representation][axis],
                                       **{f"raw_clipping_{key}": value for key, value in stats["clipping"][axis].items()}})
        for (split, map_name), indices in sorted(groups.items()):
            stats = group_statistics(indices, raw, clipped, normalized, below, above, flags)
            map_rows.append({"split": split, "map": map_name, "num_samples": len(indices),
                             "num_trajectories": len({samples[index]["traj_rel_dir"] for index in indices}),
                             "land_positive": stats["land"]["positive"], "land_negative": stats["land"]["negative"],
                             "last": stats["land"]["last"], "penultimate": stats["land"]["penultimate"],
                             "both_flags": stats["land"]["both_flags"], "any_axis_clipped": stats["any_axis_clipped"],
                             **{f"{axis}_clipped": stats["clipping"][axis]["count"] for axis in ACTION_ORDER[:3]}})
            chosen = select_examples(indices, raw, below, above, flags)
            summary["audit_examples"]["categories_per_map"][f"{split}/{map_name}"] = {
                "num_samples": len(chosen), "sample_ids": list(chosen),
                "categories": sorted({category for categories in chosen.values() for category in categories})}
            for sample_id, categories in chosen.items():
                sample = samples[sample_id]
                camera_paths = {name: Path(sample["traj_rel_dir"]) / folder / sample["img_name"]
                                for name, folder in (("front", "frontcamera"), ("down", "downcamera"))}
                examples.append({"sample_id": sample_id, "sample_id_definition": "zero_based_full_original_manifest_index",
                                 "split": split, "map": map_name, "traj_rel_dir": sample["traj_rel_dir"],
                                 "img_name": sample["img_name"],
                                 **{f"{name}_image_relative": str(path) for name, path in camera_paths.items()},
                                 **{f"{name}_image_path": str(data_root / path) for name, path in camera_paths.items()},
                                 "instruction": sample["instruction"], "raw_label": sample["label"],
                                 "action_order": list(ACTION_ORDER), "physical_4d": raw[sample_id].tolist(),
                                 "clipped_4d": clipped[sample_id].tolist(), "normalized_4d": normalized[sample_id].tolist(),
                                 "is_last_step": bool(flags[sample_id, 0]), "is_penultimate": bool(flags[sample_id, 1]),
                                 "land_rule": LAND_RULE,
                                 "clipping_axes": [axis for j, axis in enumerate(ACTION_ORDER)
                                                   if below[sample_id, j] or above[sample_id, j]],
                                 "selection_reasons": categories})
        for (split, map_name, trajectory), indices in sorted(trajectory_indices.items()):
            ids = np.asarray(indices, dtype=np.int64)
            trajectory_rows.append({"split": split, "map": map_name, "traj_rel_dir": trajectory,
                                    "num_samples": len(indices), "first_sample_id": indices[0], "last_sample_id": indices[-1],
                                    "land_positive": int(raw[ids, 3].sum()), "land_negative": int((raw[ids, 3] == 0).sum()),
                                    "last": int(flags[ids, 0].sum()), "penultimate": int(flags[ids, 1].sum()),
                                    "both_flags": int(flags[ids].all(axis=1).sum()),
                                    "any_axis_clipped": int((below[ids] | above[ids]).any(axis=1).sum())})
        summary["audit_examples"]["num_samples"] = len(examples)
        write_json(output_dir / "source_manifest.json", sources)
        write_csv(output_dir / "axis_statistics.csv", stats_rows)
        write_csv(output_dir / "map_distribution.csv", map_rows)
        write_csv(output_dir / "trajectory_distribution.csv", trajectory_rows)
        write_csv(output_dir / "samples.csv", examples)
        with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False, allow_nan=False) + "\n")
        (output_dir / "README.md").write_text(markdown_summary(summary, map_rows), encoding="utf-8")
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "audit_errors.json", {"status": "passed", "errors": []})
        return summary
    except (AuditValidationError, ValueError, KeyError, OSError, TypeError) as exc:
        errors = exc.errors if isinstance(exc, AuditValidationError) else [{"error": str(exc)}]
        failure = {"status": "failed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "errors": errors, "note": "Any previous example files are stale; this audit did not pass."}
        write_json(output_dir / "audit_errors.json", failure)
        write_json(output_dir / "summary.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-json", default="data/aerovla_train_dataset.json")
    parser.add_argument("--split-json", default="data/continuous_split_seed42.json")
    parser.add_argument("--output-dir", default=f"{DEFAULT_RUN}/supervision")
    parser.add_argument("--run-config", help="Defaults to the output directory's parent/run_config.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summary = audit(args.data_json, args.split_json, args.output_dir, args.run_config)
    print(json.dumps({"status": summary["status"], "output_dir": str(Path(args.output_dir).resolve()),
                      "split_counts": {split: stats["num_samples"] for split, stats in summary["splits"].items()},
                      "audit_examples": summary["audit_examples"]["num_samples"],
                      "run_config_match": summary["sources"]["run_config_match"]["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
