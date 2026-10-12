"""Capture complete training log lines once, then rebuild the trend on CPU.

First run: python build_training_trend.py --capture-from /path/to/run
Rebuild:   python build_training_trend.py
The original weekly report and its step-59019 snapshot remain independent.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "snapshot"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def complete_lines(path):
    payload = path.read_bytes()
    end = payload.rfind(b"\n") + 1
    return payload[:end], len(payload) - end


def capture(run_dir):
    # Read validation first so its updates already exist in the later train read.
    payloads, ignored = {}, {}
    for name in ("validation_metrics.jsonl", "metrics.jsonl"):
        payloads[name], ignored[name] = complete_lines(run_dir / name)
    payloads["run_config.json"] = (run_dir / "run_config.json").read_bytes()
    rows = [json.loads(line) for line in payloads["metrics.jsonl"].splitlines()]
    assert rows, "No complete training records"
    manifest = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "source_run": str(run_dir),
        "cutoff_step": rows[-1]["step"],
        "bin_size": 250,
        "ignored_incomplete_tail_bytes": ignored,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(),
                         "bytes": len(data), "source": str(run_dir / name)}
                  for name, data in payloads.items()},
    }
    SNAPSHOT.mkdir(exist_ok=False)
    for name, data in payloads.items():
        (SNAPSHOT / name).write_bytes(data)
    write_json(SNAPSHOT / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-from", type=Path)
    args = parser.parse_args()
    if args.capture_from:
        capture(args.capture_from.resolve())
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == info["sha256"], name
    train = read_rows(SNAPSHOT / "metrics.jsonl")
    validation = read_rows(SNAPSHOT / "validation_metrics.jsonl")
    config = json.loads((SNAPSHOT / "run_config.json").read_text())
    cutoff = manifest["cutoff_step"]
    assert [r["step"] for r in train] == list(range(1, cutoff + 1))
    values = np.array([[r[k] for k in ("loss", "lr", "grad_norm", "samples")] for r in train])
    assert np.isfinite(values).all()
    assert (values[:, [0, 2, 3]] > 0).all() and (values[:, 1] >= 0).all()
    steps_per_epoch = math.ceil(math.ceil(config["train_split"]["num_samples"] / config["batch_size"])
                                / config["gradient_accumulation_steps"])
    assert all(r["epoch"] == (r["step"] - 1) // steps_per_epoch + 1 for r in train)
    assert len({r["step"] for r in validation}) == len(validation)
    assert [r["step"] for r in validation] == sorted(r["step"] for r in validation)
    for row in validation:
        assert row["step"] == row["epoch"] * steps_per_epoch <= cutoff
        assert row["num_samples"] == config["val_split"]["num_samples"]
        assert np.isfinite(row["noise_mse"]) and row["noise_mse"] > 0
    assert any(r["epoch"] == 3 for r in validation), "Epoch 3 validation is not complete"

    width = manifest["bin_size"]
    bins = [train[i:i + width] for i in range(0, len(train), width)]
    centers = [(rows[0]["step"] + rows[-1]["step"]) / 2 for rows in bins]
    stats = []
    for rows in bins:
        losses = np.array([r["loss"] for r in rows])
        grads = np.array([r["grad_norm"] for r in rows])
        stats.append([losses.mean(), *np.quantile(losses, [.1, .9]),
                      np.median(grads), np.quantile(grads, .95)])
    stats = np.array(stats)
    boundaries = list(range(steps_per_epoch, cutoff + 1, steps_per_epoch))
    stamp = datetime.fromisoformat(manifest["captured_utc"]).astimezone(ZoneInfo("Asia/Shanghai"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120})
    blue, orange = "#3478B3", "#E38D35"
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].fill_between(centers, stats[:, 1], stats[:, 2], color=blue, alpha=.18,
                         label="Within-bin P10-P90")
    axes[0].plot(centers, stats[:, 0], color=blue, label="Train noise loss: 250-update mean")
    axes[0].scatter([v["step"] for v in validation], [v["noise_mse"] for v in validation],
                    color=orange, s=65, zorder=3, label="Full validation noise MSE")
    for val in validation:
        axes[0].annotate(f"E{val['epoch']}: {val['noise_mse']:.5f}",
                         (val["step"], val["noise_mse"]), xytext=(0, -20),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(yscale="log", ylabel="Noise MSE (log scale)",
                title=f"AeroVLA diffusion head | snapshot at update {cutoff:,}\n"
                      f"{len(validation)} epochs validated | {stamp:%Y-%m-%d %H:%M:%S} (UTC+8)")
    axes[0].legend(loc="upper right")
    indices = sorted(set(range(0, len(train), max(1, len(train) // 3000))) | {len(train) - 1})
    axes[1].plot([train[i]["step"] for i in indices], [train[i]["lr"] for i in indices], color=blue)
    axes[1].set_ylabel("Learning rate")
    axes[2].plot(centers, stats[:, 3], color=blue, label="Median per 250 updates")
    axes[2].plot(centers, stats[:, 4], color=orange, label="P95 per 250 updates")
    axes[2].axhline(config["max_grad_norm"], linestyle="--", color="#555555",
                    label=f"Clipping threshold = {config['max_grad_norm']:g}")
    axes[2].set(yscale="log", ylabel="Pre-clip gradient norm", xlabel="Optimizer update")
    axes[2].legend(loc="upper right")
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary, linestyle=":", color="#888888", alpha=.65)
        axis.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"training_trend.{suffix}", **({"dpi": 160} if suffix == "png" else {}))
    plt.close(fig)

    epoch2, epoch3 = [next(v for v in validation if v["epoch"] == epoch) for epoch in (2, 3)]
    summary = {
        "captured_utc": manifest["captured_utc"], "captured_local": stamp.isoformat(),
        "cutoff_step": cutoff, "current_epoch": train[-1]["epoch"],
        "total_optimizer_steps": steps_per_epoch * config["epochs"],
        "steps_per_epoch": steps_per_epoch, "epoch_boundaries": boundaries,
        "validation_points": [{k: v[k] for k in ("epoch", "step", "num_samples", "noise_mse")}
                              for v in validation],
        "epoch2_to_epoch3_noise_mse_reduction_percent": (1 - epoch3["noise_mse"] / epoch2["noise_mse"]) * 100,
        "epoch2_to_epoch3_land_f1_percentage_points": (epoch3["land"]["f1"] - epoch2["land"]["f1"]) * 100,
        "all_train_records_finite_contiguous": True,
        "bin_size": width, "last_bin_updates": len(bins[-1]),
        "band_meaning": "Within-bin P10-P90; not a confidence interval",
        "gradient_norm_meaning": "Before gradient clipping",
        "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ("training_trend.png", "training_trend.pdf", "build_training_trend.py")},
    }
    write_json(ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
