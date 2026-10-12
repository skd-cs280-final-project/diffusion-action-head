"""Read-only, CPU-only supervision of an already running diffusion training job.

The source JSONL files do not contain timestamps.  Throughput therefore describes
only progress observed by this process; historical per-step times are never made
up.  CSVs retain all records, while plots may be downsampled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time


TERMINAL_STATES = {"completed", "failed"}
TRAIN_FIELDS = ["step", "epoch", "loss", "lr", "grad_norm", "samples"]
VAL_FIELDS = ["step", "epoch", "split", "num_samples", "noise_mse",
              "normalized_continuous_mse"]
CHECKPOINT_FIELDS = ["directory", "global_step", "weights_exists", "weights_size_bytes",
                     "weights_modified_utc", "config_exists", "config_readable", "files_present"]


def utc_iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def flatten(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def is_finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_write(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path, records, preferred_fields=()):
    fields = list(dict.fromkeys([*preferred_fields, *(key for row in records for key in row)]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in records:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                         else value for key, value in row.items()})
    atomic_write(path, stream.getvalue())


class JsonlTail:
    """Incrementally read complete lines, including a partial UTF-8 trailing line."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.line_number = 0
        self.identity = None
        self.rows = []
        self.warnings = []
        self.mtime = None
        self.generation = 0

    def poll(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        identity = (stat.st_dev, stat.st_ino)
        if (self.identity is not None and self.identity != identity) or stat.st_size < self.offset:
            self.offset = self.line_number = 0
            self.pending = b""
            self.rows.clear()
            self.warnings.clear()
            self.generation += 1
        self.identity = identity
        self.mtime = stat.st_mtime
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            content = self.pending + stream.read()
            self.offset = stream.tell()
        lines = content.split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            self.line_number += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL record must be an object")
            except (ValueError, UnicodeDecodeError) as error:
                self.warnings.append({"source": self.path.name, "line": self.line_number,
                                      "kind": "invalid_json", "field": "", "value": str(error)})
                continue
            flat = flatten(row)
            self.rows.append(flat)
            for field, value in flat.items():
                if isinstance(value, float) and not math.isfinite(value):
                    self.warnings.append({"source": self.path.name, "line": self.line_number,
                                          "kind": "nonfinite", "field": field, "value": str(value)})


def read_json_object(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError, UnicodeDecodeError):
        # The launcher may be writing status.json while a poll is taking place.
        return {}


def status_name(status):
    if status.get("failed") is True:
        return "failed"
    if status.get("completed") is True:
        return "completed"
    return str(status.get("status", "waiting")).lower()


def training_schedule(config):
    try:
        samples = int(config["train_split"]["num_samples"])
        if config.get("max_samples") is not None:
            samples = min(samples, int(config["max_samples"]))
        batch = int(config["batch_size"])
        accumulation = int(config["gradient_accumulation_steps"])
        epochs = int(config["epochs"])
        if min(samples, batch, accumulation, epochs) < 1:
            return None, None
        per_epoch = math.ceil(math.ceil(samples / batch) / accumulation)
        total = per_epoch * epochs
        if config.get("max_steps") is not None:
            total = min(total, int(config["max_steps"]))
        return per_epoch, total
    except (KeyError, ValueError, TypeError):
        return None, None


def checkpoint_inventory(run_dir):
    """Inspect small configs and weight file metadata, never load model weights."""
    run_dir = Path(run_dir)
    directories = [run_dir]
    directories.extend(sorted(path for path in run_dir.iterdir()
                              if (path.name == "best" or path.name.startswith("checkpoint-")) and path.is_dir()))
    records = []
    for directory in directories:
        weights = directory / "diffusion_head.pt"
        config_path = directory / "diffusion_config.json"
        try:
            stat = weights.stat() if weights.is_file() else None
        except FileNotFoundError:
            stat = None
        config_exists = config_path.is_file()
        if directory == run_dir and stat is None and not config_exists:
            continue
        config = read_json_object(config_path) if config_exists else {}
        records.append({"directory": "." if directory == run_dir else directory.name,
                        "global_step": config.get("training_metadata", {}).get("global_step"),
                        "weights_exists": stat is not None,
                        "weights_size_bytes": stat.st_size if stat is not None else None,
                        "weights_modified_utc": utc_iso(stat.st_mtime) if stat is not None else None,
                        "config_exists": config_exists, "config_readable": bool(config),
                        "files_present": stat is not None and config_exists})
    return records


def rolling_mean(values, window=100):
    result, total, count = [], 0.0, 0
    for index, value in enumerate(values):
        if is_finite_number(value):
            total += value
            count += 1
        if index >= window and is_finite_number(values[index - window]):
            total -= values[index - window]
            count -= 1
        result.append(total / count if count else float("nan"))
    return result


def plot_curves(path, train, validation, max_points=5000, clip_norm=None):
    # Importing torch or reading model weights is deliberately unnecessary here.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(path).parent / ".cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    stride = max(1, math.ceil(len(train) / max_points))
    selected = list(range(0, len(train), stride))
    if train and (not selected or selected[-1] != len(train) - 1):
        selected.append(len(train) - 1)

    def values(rows, key):
        return [row.get(key) if is_finite_number(row.get(key)) else float("nan") for row in rows]

    steps = values(train, "step")
    selected_steps = [steps[index] for index in selected]
    losses = values(train, "loss")
    smoothed = rolling_mean(losses)
    axes[0, 0].plot(selected_steps, [losses[i] for i in selected], alpha=0.35, label="Loss")
    axes[0, 0].plot(selected_steps, [smoothed[i] for i in selected], label="Rolling mean (100 updates)")
    axes[0, 0].set_title("Training diffusion noise loss")
    learning_rates = values(train, "lr")
    axes[0, 1].plot(selected_steps, [learning_rates[i] for i in selected], label="Learning rate")
    axes[0, 1].set_title("Learning rate")
    gradients = values(train, "grad_norm")
    axes[1, 0].plot(selected_steps, [gradients[i] for i in selected], label="Before clipping")
    if is_finite_number(clip_norm):
        axes[1, 0].axhline(clip_norm, color="red", linestyle="--", label="Clip threshold")
    axes[1, 0].set_title("Gradient norm")
    val_steps = values(validation, "step")
    for field, label in [("noise_mse", "Noise MSE"), ("normalized_continuous_mse", "Normalized action MSE")]:
        axes[1, 1].plot(val_steps, values(validation, field), marker="o", label=label)
    axes[1, 1].set_title("Validation diffusion and action error")
    for name in ("fwd", "down", "yaw"):
        axes[2, 0].plot(val_steps, values(validation, f"physical.{name}.mae"), marker="o", label=name)
    axes[2, 0].set_title("Validation action MAE (native units; yaw in radians)")
    for name in ("precision", "recall", "f1", "accuracy"):
        axes[2, 1].plot(val_steps, values(validation, f"land.{name}"), marker="o", label=name)
    axes[2, 1].set_title("Validation LAND classification")
    axes[2, 1].set_ylim(-0.05, 1.05)
    for axis in axes.flat:
        axis.set_xlabel("Optimizer update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    descriptor, temporary = tempfile.mkstemp(prefix=".curves.", suffix=".png", dir=Path(path).parent)
    os.close(descriptor)
    try:
        figure.savefig(temporary, dpi=130, format="png")
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.unlink(temporary)


class TrainingMonitor:
    def __init__(self, run_dir, log_dir, *, plot_interval=60, make_plots=True):
        self.run_dir, self.log_dir = Path(run_dir), Path(log_dir)
        self.output_dir = self.run_dir / "monitoring"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train = JsonlTail(self.run_dir / "metrics.jsonl")
        self.validation = JsonlTail(self.run_dir / "validation_metrics.jsonl")
        self.plot_interval, self.make_plots = plot_interval, make_plots
        self.last_plot = None
        self.plot_warning = None
        self.baseline = None
        self.rate = None
        self.config, self.status = {}, {}

    def refresh(self, *, monotonic_now=None, wall_now=None):
        monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
        wall_now = time.time() if wall_now is None else wall_now
        self.train.poll()
        self.validation.poll()
        self.config = read_json_object(self.run_dir / "run_config.json") or self.config
        self.status = read_json_object(self.log_dir / "status.json") or self.status
        state = status_name(self.status)
        terminal = state in TERMINAL_STATES
        if terminal:
            # The launcher may finish between the first JSONL poll and status read.
            self.train.poll()
            self.validation.poll()
        train, validation = self.train.rows, self.validation.rows
        checkpoints = checkpoint_inventory(self.run_dir)
        latest = train[-1] if train else {}
        step = latest.get("step")
        epoch = latest.get("epoch")
        per_epoch, total = training_schedule(self.config)
        cumulative_samples = sum(row["samples"] for row in train if is_finite_number(row.get("samples")))
        signature = (epoch, len(validation), self.train.generation, self.validation.generation)
        if is_finite_number(step):
            if self.baseline is None or self.baseline["signature"] != signature or step < self.baseline["step"]:
                self.baseline = {"step": step, "samples": cumulative_samples, "time": monotonic_now,
                                 "signature": signature}
                self.rate = None
            elif step > self.baseline["step"] and monotonic_now > self.baseline["time"]:
                # Freeze the rate while a step is stationary (e.g. epoch validation).
                if self.rate is None or step != self.rate["last_observed_step"]:
                    elapsed = monotonic_now - self.baseline["time"]
                    self.rate = {"optimizer_steps_per_second": (step - self.baseline["step"]) / elapsed,
                                 "samples_per_second": (cumulative_samples - self.baseline["samples"]) / elapsed,
                                 "observed_window_seconds": elapsed,
                                 "first_observed_step": self.baseline["step"], "last_observed_step": step}
        eta = None
        if self.rate and total is not None and is_finite_number(step):
            eta = max(0, total - step) / self.rate["optimizer_steps_per_second"]
        last_validation_step = validation[-1].get("step") if validation else None
        validation_due = (is_finite_number(step) and step > 0 and
                          ((per_epoch is not None and step % per_epoch == 0) or step == total or
                           (self.config.get("eval_steps", 0) > 0 and step % self.config["eval_steps"] == 0)))
        phase = state if terminal else ("validation_or_checkpoint" if validation_due and last_validation_step != step
                                        else "training" if train else "waiting_for_training_metrics")
        losses = [row["loss"] for row in train[-100:] if is_finite_number(row.get("loss"))]
        warnings = [*self.train.warnings, *self.validation.warnings]
        if self.make_plots and (self.last_plot is None or terminal or monotonic_now - self.last_plot >= self.plot_interval):
            try:
                plot_curves(self.output_dir / "training_curves.png", train, validation,
                            clip_norm=self.config.get("max_grad_norm"))
                self.plot_warning = None
            except Exception as error:
                # Monitoring CSV/JSON must survive missing or broken plotting packages.
                self.plot_warning = f"{type(error).__name__}: {error}"
            self.last_plot = monotonic_now
        if self.plot_warning:
            warnings.append({"source": "monitor", "kind": "plot_unavailable", "field": "", "line": "",
                             "value": self.plot_warning})
        summary = json_safe({
            "updated_utc": utc_iso(wall_now), "status": state, "phase": phase, "terminal": terminal,
            "run_dir": str(self.run_dir.resolve()), "log_dir": str(self.log_dir.resolve()),
            "launcher_status": self.status, "config_available": bool(self.config),
            "training_records": len(train), "validation_records": len(validation),
            "latest_training": latest, "latest_validation": validation[-1] if validation else None,
            "loss_mean_last_100_records": sum(losses) / len(losses) if losses else None,
            "total_optimizer_steps": total, "optimizer_steps_per_epoch": per_epoch,
            "progress_fraction": min(1, step / total) if total and is_finite_number(step) else None,
            "observed_training_samples": cumulative_samples, "throughput": self.rate,
            "estimated_remaining_training_seconds": eta,
            "estimated_training_finish_utc": utc_iso(wall_now + eta) if eta is not None else None,
            "eta_includes_validation": False,
            "timing_note": "Only steps observed by this monitor are timed with monotonic time. "
                           "Rate resets after an epoch change or validation result. Training ETA excludes future "
                           "validation; observed rates may include saving, data loading and polling overhead. "
                           "Source records have no per-step timestamps.",
            "train_log_modified_utc": utc_iso(self.train.mtime) if self.train.mtime is not None else None,
            "train_log_age_seconds": max(0, wall_now - self.train.mtime) if self.train.mtime is not None else None,
            "pending_training_tail_bytes": len(self.train.pending),
            "pending_validation_tail_bytes": len(self.validation.pending),
            "warning_count": len(warnings), "recent_warnings": warnings[-50:],
            "checkpoint_records": len(checkpoints), "checkpoints": checkpoints,
            "checkpoint_note": "files_present only means the weight and config paths exist; "
                               "it does not verify weight contents or prove an in-progress save has finished.",
            "artifacts": {"train_csv": "train_metrics.csv", "validation_csv": "validation_metrics.csv",
                          "warnings_csv": "warnings.csv", "checkpoints_csv": "checkpoints.csv",
                          "curves": "training_curves.png"},
        })
        write_csv(self.output_dir / "train_metrics.csv", train, TRAIN_FIELDS)
        write_csv(self.output_dir / "validation_metrics.csv", validation, VAL_FIELDS)
        write_csv(self.output_dir / "warnings.csv", warnings, ["source", "line", "kind", "field", "value"])
        write_csv(self.output_dir / "checkpoints.csv", checkpoints, CHECKPOINT_FIELDS)
        atomic_write(self.output_dir / "latest_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_write(self.output_dir / "status.md", self.markdown(summary))
        return summary

    @staticmethod
    def markdown(summary):
        latest = summary["latest_training"]
        rate = summary["throughput"]
        eta = summary["estimated_remaining_training_seconds"]
        lines = ["# 扩散头训练监督", "", f"更新时间（UTC）：{summary['updated_utc']}", "",
                 "| 项目 | 当前值 |", "| --- | --- |",
                 f"| 状态 / 阶段 | {summary['status']} / {summary['phase']} |",
                 f"| epoch | {latest.get('epoch', '等待')} |",
                 f"| optimizer step / 计划总数 | {latest.get('step', '等待')} / {summary['total_optimizer_steps']} |",
                 f"| 最新 loss / 最近 100 条均值 | {latest.get('loss')} / {summary['loss_mean_last_100_records']} |",
                 f"| 最新学习率 / 裁剪前梯度范数 | {latest.get('lr')} / {latest.get('grad_norm')} |",
                 f"| 训练 / 验证记录数 | {summary['training_records']} / {summary['validation_records']} |",
                 f"| 检查点目录记录数 | {summary['checkpoint_records']} |",
                 f"| 训练日志最后修改（UTC） | {summary['train_log_modified_utc']} |",
                 f"| 距训练日志更新（秒） | {summary['train_log_age_seconds']} |",
                 f"| 告警数 | {summary['warning_count']} |", ""]
        if rate:
            lines.append(f"监控窗口观测吞吐：{rate['optimizer_steps_per_second']:.5f} optimizer step/s，"
                         f"{rate['samples_per_second']:.3f} samples/s；窗口 {rate['observed_window_seconds']:.1f} 秒，"
                         f"step {rate['first_observed_step']} → {rate['last_observed_step']}。")
        else:
            lines.append("吞吐与 ETA：等待本监控进程在同一训练阶段观测到至少两次不同 step。")
        if eta is not None:
            lines.extend(["", f"预计剩余训练计算时间：{eta / 3600:.2f} 小时（**不含后续全量验证时间**）。"])
        lines.extend(["", "计时使用本监控进程的 monotonic 观测时间；跨 epoch 或出现新验证结果时重置窗口。"
                      "吞吐可能包含保存、数据加载与轮询误差。原始训练记录没有逐步时间戳，未补造历史 wall time。"
                      "验证期间训练日志停止更新属于正常现象。", "",
                      "- [完整训练 CSV](train_metrics.csv)：保留所有训练记录和字段。",
                      "- [完整验证 CSV](validation_metrics.csv)：嵌套字段使用点号展开，包含动作误差与 LAND 指标。",
                      "- [告警 CSV](warnings.csv)：坏行与非有限值，原始 JSONL 保留不改。",
                      "- [检查点 CSV](checkpoints.csv)：仅检查 root、best、checkpoint-* 的文件存在性、大小、修改时间和 config step。"
                      "`files_present` 不代表权重内容已校验，也不保证正在写入的保存已完成。",
                      "- [机器可读摘要](latest_summary.json)", "- [训练与验证曲线](training_curves.png)", ""])
        if summary["latest_validation"]:
            lines.extend(["## 最近一次验证", "", "| 指标 | 值 |", "| --- | --- |"])
            for key, value in summary["latest_validation"].items():
                lines.append(f"| {key} | {value} |")
            lines.append("")
        if summary["recent_warnings"]:
            lines.extend(["## 最近告警（最多 50 条）", "", "| 来源 | 行 | 类型 | 字段 | 内容 |",
                          "| --- | --- | --- | --- | --- |"])
            for warning in summary["recent_warnings"]:
                values = [str(warning.get(key, "")).replace("|", "\\|").replace("\n", " ")
                          for key in ("source", "line", "kind", "field", "value")]
                lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--plot-interval", type=float, default=60)
    parser.add_argument("--once", action="store_true", help="Export one snapshot and exit")
    parser.add_argument("--no-plots", action="store_true", help="Only export CSV, JSON and Markdown")
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.plot_interval <= 0:
        parser.error("intervals must be positive")
    monitor = TrainingMonitor(args.run_dir, args.log_dir, plot_interval=args.plot_interval,
                              make_plots=not args.no_plots)
    while True:
        summary = monitor.refresh()
        print(json.dumps({"updated_utc": summary["updated_utc"], "status": summary["status"],
                          "step": summary["latest_training"].get("step"),
                          "warnings": summary["warning_count"]}), flush=True)
        if args.once or summary["terminal"]:
            return 1 if summary["status"] == "failed" else 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
