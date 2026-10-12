"""Rebuild this weekly report's statistics and figures from its frozen snapshot.

Run with the aero_vla Python environment. This script never loads a model or
reads live training metrics; snapshot/manifest.json defines the report cutoff.
"""
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "snapshot"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def read_json(name):
    return json.loads((SNAPSHOT / name).read_text())


def read_jsonl(name):
    return [json.loads(line) for line in (SNAPSHOT / name).read_text().splitlines() if line]


def statistics(rows):
    losses = np.array([row["loss"] for row in rows])
    gradients = np.array([row["grad_norm"] for row in rows])
    samples = np.array([row["samples"] for row in rows])
    return {
        "first_step": rows[0]["step"], "last_step": rows[-1]["step"],
        "updates": len(rows), "sample_visits": int(samples.sum()),
        "loss_mean": float(losses.mean()), "loss_sample_weighted": float(np.average(losses, weights=samples)),
        "loss_std_population": float(losses.std()),
        "loss_p10": float(np.quantile(losses, 0.1)), "loss_p90": float(np.quantile(losses, 0.9)),
        "grad_norm_median": float(np.median(gradients)), "grad_norm_p95": float(np.quantile(gradients, 0.95)),
        "grad_norm_max": float(gradients.max()), "preclip_grad_norm_above_1": int((gradients > 1).sum()),
    }


manifest = read_json("manifest.json")
for name, info in manifest["files"].items():
    assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == info["sha256"], name
train, validation = read_jsonl("metrics.jsonl"), read_jsonl("validation_metrics.jsonl")
monitor, config, supervision = read_json("monitor_summary.json"), read_json("run_config.json"), read_json("supervision_summary.json")
cutoff = manifest["cutoff_step"]
assert [row["step"] for row in train] == list(range(1, cutoff + 1))
assert all(np.isfinite(row[key]) for row in train for key in ("loss", "lr", "grad_norm", "samples"))
assert len(validation) == 2 and all(row["num_samples"] == 41176 for row in validation)
assert read_json("best_metrics.json") == min(validation, key=lambda row: row["noise_mse"])
epochs = {str(epoch): statistics([row for row in train if row["epoch"] == epoch])
          for epoch in sorted({row["epoch"] for row in train})}
windows = {"first_1000": statistics(train[:1000]), "last_100": statistics(train[-100:]),
           "last_1000": statistics(train[-1000:])}
baseline = {"method": "Predict train split clipped target mean; derive validation MSE from population variance and mean difference.",
            "physical": {}, "normalized_coordinate_mse": {}}
for axis in ("fwd", "down", "yaw"):
    for representation, destination in (("clipped_4d_float32", "physical"), ("normalized_4d_float32", "normalized_coordinate_mse")):
        tr = supervision["splits"]["train"][representation][axis]
        va = supervision["splits"]["val"][representation][axis]
        mse = va["std_population"] ** 2 + (va["mean"] - tr["mean"]) ** 2
        baseline[destination][axis] = {"prediction": tr["mean"], "mse": mse, "rmse": mse ** 0.5}
baseline["normalized_continuous_mse"] = sum(v["mse"] for v in baseline["normalized_coordinate_mse"].values()) / 3
baseline["always_continue_accuracy"] = validation[-1]["land"]["support_negative"] / validation[-1]["num_samples"]
improvements = {}
for key in ("noise_mse", "normalized_continuous_mse"):
    improvements[key] = (1 - validation[1][key] / validation[0][key]) * 100
for axis in ("fwd", "down", "yaw"):
    for metric in ("mae", "rmse"):
        improvements[f"{axis}.{metric}"] = (1 - validation[1]["physical"][axis][metric] / validation[0]["physical"][axis][metric]) * 100
for metric in ("precision", "recall", "f1", "accuracy"):
    improvements[f"land.{metric}.percentage_points"] = (validation[1]["land"][metric] - validation[0]["land"][metric]) * 100

# Bound validation pauses using adjacent monitor observations, not invented
# per-step timestamps. The bounds also include any checkpoint/loader overhead.
observations = read_jsonl("monitor_observations.jsonl")
pauses = []
for result in validation:
    indices = [i for i, row in enumerate(observations) if row["step"] == result["step"]]
    if not indices or indices[0] == 0 or indices[-1] + 1 >= len(observations):
        continue
    first, last = indices[0], indices[-1]
    stamp = lambda i: datetime.fromisoformat(observations[i]["updated_utc"])
    pauses.append({"epoch": result["epoch"], "step": result["step"],
                   "first_boundary_observed_utc": observations[first]["updated_utc"],
                   "last_boundary_observed_utc": observations[last]["updated_utc"],
                   "next_progress_observed_utc": observations[last + 1]["updated_utc"],
                   "lower_minutes": (stamp(last) - stamp(first)).total_seconds() / 60,
                   "upper_minutes": (stamp(last + 1) - stamp(first - 1)).total_seconds() / 60})
now = datetime.fromisoformat(manifest["cutoff_utc"])
remaining_training = monitor["estimated_remaining_training_seconds"]
remaining_validations = config["epochs"] - len(validation)
pause_low = min(p["lower_minutes"] for p in pauses) * 60
pause_high = max(p["upper_minutes"] for p in pauses) * 60
eta = {"remaining_training_hours": remaining_training / 3600,
       "remaining_validations": remaining_validations,
       "validation_pause_bounds_minutes": [pause_low / 60, pause_high / 60],
       "estimated_remaining_total_hours": [(remaining_training + remaining_validations * p) / 3600 for p in (pause_low, pause_high)],
       "estimated_finish_utc": [(now + timedelta(seconds=remaining_training + remaining_validations * p)).isoformat() for p in (pause_low, pause_high)],
       "note": "Extrapolation assuming current training throughput and observed epoch-end pause range; not a confidence interval or guarantee."}
analysis = {"cutoff_step": cutoff, "cutoff_utc": manifest["cutoff_utc"], "progress": cutoff / monitor["total_optimizer_steps"],
            "elapsed_hours": (now - datetime.fromisoformat(read_json("launcher_status.json")["started_utc"].replace("Z", "+00:00"))).total_seconds() / 3600,
            "epochs": epochs, "windows": windows, "all_records_finite_contiguous": True,
            "max_gradient_record": max(train, key=lambda row: row["grad_norm"]),
            "validation": validation, "epoch1_to_epoch2_improvement_percent": improvements,
            "constant_baseline": baseline, "observed_validation_pauses": pauses, "eta": eta}
(ROOT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
COLORS = ["#3478B3", "#E38D35"]
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
bins = [train[i:i + 250] for i in range(0, len(train), 250)]
mid = [(rows[0]["step"] + rows[-1]["step"]) / 2 for rows in bins]
averages = [statistics(rows) for rows in bins]
axes[0].fill_between(mid, [v["loss_p10"] for v in averages], [v["loss_p90"] for v in averages], color=COLORS[0], alpha=0.18, label="Within-bin P10-P90")
axes[0].plot(mid, [v["loss_mean"] for v in averages], color=COLORS[0], label="Train noise loss: 250-update mean")
axes[0].scatter([v["step"] for v in validation], [v["noise_mse"] for v in validation], color=COLORS[1], s=65, zorder=3, label="Full validation noise MSE")
axes[0].set(yscale="log", ylabel="Noise MSE (log scale)", title=f"AeroVLA diffusion head | snapshot at update {cutoff:,}")
axes[0].legend()
stride = max(1, len(train) // 3000)
axes[1].plot([r["step"] for r in train[::stride]], [r["lr"] for r in train[::stride]], color=COLORS[0])
axes[1].set_ylabel("Learning rate")
axes[2].plot(mid, [v["grad_norm_median"] for v in averages], label="Median per 250 updates", color=COLORS[0])
axes[2].plot(mid, [v["grad_norm_p95"] for v in averages], label="P95 per 250 updates", color=COLORS[1])
axes[2].axhline(1, linestyle="--", color="#555555", label="Clipping threshold = 1")
axes[2].set(yscale="log", ylabel="Pre-clip gradient norm", xlabel="Optimizer update")
axes[2].legend()
for axis in axes:
    for boundary in (23080, 46160):
        axis.axvline(boundary, linestyle=":", color="#888888", alpha=.65)
    axis.grid(alpha=.2)
fig.savefig(ROOT / "training_trend.png", dpi=160)
fig.savefig(ROOT / "training_trend.pdf")
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
for col, name in enumerate(("fwd", "down", "yaw")):
    for row, metric in enumerate(("mae", "rmse")):
        axis = axes[row, col]
        values = [v["physical"][name][metric] for v in validation]
        labels, colors = ["Epoch 1", "Epoch 2"], COLORS
        if metric == "rmse":
            values = [baseline["physical"][name]["rmse"], *values]
            labels, colors = ["Train-mean\nconstant", *labels], ["#AAAAAA", *COLORS]
        bars = axis.bar(labels, values, color=colors, width=.55)
        axis.bar_label(bars, fmt="%.4f", padding=3)
        axis.set_title(f"{name.upper()} {metric.upper()}")
        axis.set_ylabel("Radians" if name == "yaw" else "Original distance unit")
        axis.set_ylim(0, max(values) * 1.2)
        axis.grid(axis="y", alpha=.2)
fig.suptitle("Full validation: 41,176 frames; targets clipped to training ranges\nConstant baseline derived analytically from train means and validation moments")
fig.savefig(ROOT / "validation_actions.png", dpi=160)
fig.savefig(ROOT / "validation_actions.pdf")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for axis, val in zip(axes[:2], validation):
    land = val["land"]
    matrix = np.array([[land["tn"], land["fp"]], [land["fn"], land["tp"]]])
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    axis.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            axis.text(j, i, f"{matrix[i,j]:,}\n{fractions[i,j]*100:.2f}% of row", ha="center", va="center", color="white" if fractions[i,j] > .5 else "black")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Continue", "LAND"], yticklabels=["Continue", "LAND"],
             xlabel="Prediction", ylabel="Ground truth", title=f"Epoch {val['epoch']} | LAND confusion")
x = np.arange(3)
for index, val in enumerate(validation):
    bars = axes[2].bar(x + (index - .5) * .34, [100 * val["land"][key] for key in ("precision", "recall", "f1")], .34, label=f"Epoch {val['epoch']}", color=COLORS[index])
    axes[2].bar_label(bars, fmt="%.2f", fontsize=8)
axes[2].set(xticks=x, xticklabels=["Precision", "Recall", "F1"], ylabel="Percent", ylim=(0, 110), title="LAND positive class (910 frames)")
axes[2].legend(loc="lower right")
axes[2].grid(axis="y", alpha=.2)
fig.savefig(ROOT / "land_validation.png", dpi=160)
fig.savefig(ROOT / "land_validation.pdf")
plt.close(fig)
print(json.dumps({"cutoff": cutoff, "epochs": epochs, "windows": windows, "eta": eta}, indent=2))
