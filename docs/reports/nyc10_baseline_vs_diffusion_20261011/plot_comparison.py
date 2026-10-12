#!/usr/bin/env python3
"""Plot the paired NYC evaluation comparison from an audited JSON snapshot.

Usage:
    python plot_comparison.py [--data comparison_data.json] [--output-dir DIR]

This script reads only comparison_data.json and writes PNG/PDF figures. It does
not load models, use CUDA, or change evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
COLORS = {"baseline": "#2878B5", "diffusion": "#E9852A"}
LABELS = {"baseline": "Historical AeroVLA", "diffusion": "Diffusion ckpt-70000"}
REFERENCE_COLOR = "#929AA3"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "comparison_data.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def validate(data):
    episodes = data["episodes"]
    if len(episodes) != 10:
        raise ValueError(f"Expected 10 matched episodes, got {len(episodes)}")
    if sorted(e["index"] for e in episodes) != list(range(1, 11)):
        raise ValueError("Episode indices must be exactly 1..10")
    if len({e["uuid"] for e in episodes}) != 10:
        raise ValueError("Duplicate episode UUID")
    for ep in episodes:
        for name, positions in [
            ("target", [ep["target"]]),
            ("reference", ep["reference"]),
            ("baseline", ep["baseline"]["positions"]),
            ("diffusion", ep["diffusion"]["positions"]),
        ]:
            if not positions:
                raise ValueError(f"{ep['uuid']}: empty {name} positions")
            for pos in positions:
                if len(pos) != 3 or not all(math.isfinite(float(v)) for v in pos):
                    raise ValueError(f"{ep['uuid']}: invalid {name} position {pos}")
        for model in COLORS:
            values = ep[model]
            if not isinstance(values["success"], bool):
                raise ValueError(f"{ep['uuid']}: success must be boolean")
            if len(values["positions"]) != 5 * values["action_count"] + 1:
                raise ValueError(
                    f"{ep['uuid']}: {model} saved positions are not 5/action + initial"
                )
            for metric in ("ne_m", "final_target_distance_m", "path_length_m", "legacy_spl"):
                if not math.isfinite(float(values[metric])):
                    raise ValueError(f"{ep['uuid']}: nonfinite {model} {metric}")
    for model in COLORS:
        for metric in ("sr", "osr", "legacy_spl", "ne_m", "action_count"):
            if not math.isfinite(float(data["aggregates"]["all"][model][metric])):
                raise ValueError(f"Nonfinite aggregate {model} {metric}")
    return sorted(episodes, key=lambda e: e["index"])


def save_figure(fig, stem, out, plt):
    for extension in ("png", "pdf"):
        path = out / f"{stem}.{extension}"
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        print(path)
    plt.close(fig)


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E8EB", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def short_label(ep):
    return f"{ep['index']:02d}\n{ep.get('short_id', ep['uuid'][:8])}"


def overview(data, episodes, out, plt, np):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.13, top=0.84, hspace=0.50, wspace=0.22)
    fig.suptitle("AeroVLA vs. diffusion action head | 10 matched NYC tasks", fontsize=21, y=0.98)
    fig.text(0.5, 0.939, "Historical baseline results; paired task comparison, not a controlled rerun",
             ha="center", fontsize=12, color="#545D66")
    width = 0.36
    models = tuple(COLORS)
    aggregate_ax = axes[0, 0]
    group_x = np.arange(3)
    for i, model in enumerate(models):
        agg = data["aggregates"]["all"][model]
        heights = [100 * agg[k] for k in ("sr", "osr", "legacy_spl")]
        bars = aggregate_ax.bar(group_x + (i - 0.5) * width, heights, width,
                                color=COLORS[model], label=LABELS[model], zorder=3)
        aggregate_ax.bar_label(bars, labels=[f"{v:.1f}%" for v in heights], padding=4, fontsize=11)
    aggregate_ax.set_xticks(group_x, ["Success (SR)", "Oracle success (OSR)", "Legacy SPL"])
    aggregate_ax.set_ylim(0, 112)
    aggregate_ax.set_yticks([0, 25, 50, 75, 100])
    aggregate_ax.set_ylabel("Rate / score (%)")
    aggregate_ax.set_title("Aggregate results across all 10 tasks", loc="left", fontweight="bold")
    style_axis(aggregate_ax)

    specifications = [
        (axes[0, 1], "ne_m", 1, "Reference-endpoint NE (lower is better)", "Distance (m)"),
        (axes[1, 0], "legacy_spl", 100, "Legacy SPL by task (higher is better)", "Legacy SPL (%)"),
        (axes[1, 1], "action_count", 1, "Executed high-level actions by task", "Actions"),
    ]
    task_x = np.arange(len(episodes))
    for ax, key, factor, title, ylabel in specifications:
        for i, model in enumerate(models):
            vals = [ep[model][key] * factor for ep in episodes]
            ax.bar(task_x + (i - 0.5) * width, vals, width, color=COLORS[model], zorder=3)
        ax.set_xticks(task_x, [short_label(ep) for ep in episodes], fontsize=8.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Matched task index / UUID prefix", labelpad=7)
        ax.set_ylim(bottom=0)
        if key == "legacy_spl":
            ax.set_ylim(0, 108)
        else:
            ax.margins(y=0.13)
        style_axis(ax)
    handles, labels = aggregate_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
               ncol=2, frameon=False, fontsize=12)
    fig.text(0.065, 0.052,
             "NE is the 3D distance to the reference trajectory endpoint. Legacy SPL follows the project evaluator.",
             fontsize=10, color="#545D66")
    fig.text(0.065, 0.027,
             "Baseline actions are inferred as (saved states - 1) / 5. Fewer actions alone do not imply better navigation.",
             fontsize=10, color="#545D66")
    save_figure(fig, "comparison_overview", out, plt)


def trajectories(episodes, out, plt, np, Line2D):
    fig, axes = plt.subplots(2, 5, figsize=(24, 11))
    fig.subplots_adjust(left=0.045, right=0.985, top=0.79, bottom=0.115, wspace=0.30, hspace=0.44)
    fig.suptitle("Matched NYC tasks | reference and executed XY trajectories", fontsize=22, y=0.975)
    fig.text(0.5, 0.931,
             "Historical AeroVLA vs. diffusion ckpt-70000; each panel uses equal X/Y scale",
             ha="center", fontsize=12, color="#545D66")
    for ax, ep in zip(axes.flat, episodes):
        reference = np.asarray(ep["reference"], dtype=float)
        target = np.asarray(ep["target"], dtype=float)
        ax.plot(reference[:, 0], reference[:, 1], color=REFERENCE_COLOR,
                linestyle="--", linewidth=1.6, alpha=0.9, zorder=1)
        for model in COLORS:
            p = np.asarray(ep[model]["positions"], dtype=float)
            ax.plot(p[:, 0], p[:, 1], color=COLORS[model], linewidth=1.8, alpha=0.92, zorder=2)
            ax.scatter(*p[0, :2], s=45, marker="o", facecolors="white", edgecolors=COLORS[model],
                       linewidths=1.7, zorder=5)
            ax.scatter(*p[-1, :2], s=60, marker="X", color=COLORS[model],
                       edgecolors="white", linewidths=0.5, zorder=6)
        ax.scatter(*target[:2], s=140, marker="*", color="#252B30",
                   edgecolors="white", linewidths=0.7, zorder=7)
        b_result = "SUCCESS" if ep["baseline"]["success"] else "FAIL"
        d_result = "SUCCESS" if ep["diffusion"]["success"] else "FAIL"
        ax.set_title(f"{ep['index']:02d} | {ep.get('short_id', ep['uuid'][:8])}\n"
                     f"Baseline {b_result}  /  Diffusion {d_result}",
                     fontsize=10.7, pad=9)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.13)
        ax.grid(color="#E5E8EB", linewidth=0.7)
        for edge in ("top", "right"):
            ax.spines[edge].set_visible(False)
        ax.text(0.02, 0.02, f"3D target distance: {ep['baseline']['final_target_distance_m']:.1f} / "
                f"{ep['diffusion']['final_target_distance_m']:.1f} m",
                transform=ax.transAxes, fontsize=8.5, color="#404850",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=2), zorder=9)
    handles = [
        Line2D([], [], color=REFERENCE_COLOR, linestyle="--", lw=1.8, label="Reference trajectory"),
        Line2D([], [], color=COLORS["baseline"], lw=2, label=LABELS["baseline"]),
        Line2D([], [], color=COLORS["diffusion"], lw=2, label=LABELS["diffusion"]),
        Line2D([], [], color="#252B30", marker="*", linestyle="none", markersize=11, label="Target"),
        Line2D([], [], color="#525A62", marker="o", markerfacecolor="white", linestyle="none",
               markersize=7, label="Each method's initial position"),
        Line2D([], [], color="#525A62", marker="X", linestyle="none", markersize=8,
               label="Each method's final position"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.90),
               ncol=3, frameon=False, fontsize=11)
    fig.text(0.045, 0.048,
             "XY projection omits altitude. Target distances are 3D; panel scales differ. "
             "Start and end markers are drawn for both methods and may overlap.",
             fontsize=10.5, color="#545D66")
    fig.text(0.045, 0.021,
             "These are existing baseline results matched by task UUID; runtime conditions and initial positions may differ.",
             fontsize=10.5, color="#545D66")
    save_figure(fig, "trajectory_comparison", out, plt)


def altitude(episodes, out, plt, np, Line2D, MaxNLocator):
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.79, bottom=0.14, wspace=0.33, hspace=0.48)
    fig.suptitle("Matched NYC tasks | altitude relative to the target", fontsize=22, y=0.98)
    fig.text(0.5, 0.932, "Height above target = target Z - vehicle Z (AirSim NED coordinates)",
             ha="center", fontsize=12, color="#545D66")
    for ax, ep in zip(axes.flat, episodes):
        target_z = ep["target"][2]
        ax.axhline(0, color="#5A626A", lw=1, linestyle="--", zorder=1)
        for model in COLORS:
            # The evaluator repeats each action's observed state five times.
            # Index 0 is the initial state; 5, 10, ... recover action boundaries,
            # not intermediate motion samples or reference-video frames.
            p = np.asarray(ep[model]["positions"], dtype=float)[::5]
            actions = np.arange(len(p))
            heights = target_z - p[:, 2]
            ax.plot(actions, heights, color=COLORS[model], lw=1.7, alpha=0.95, zorder=2)
            ax.scatter(actions[0], heights[0], s=28, marker="o", facecolors="white",
                       edgecolors=COLORS[model], linewidths=1.3, zorder=3)
            ax.scatter(actions[-1], heights[-1], s=34, marker="X", color=COLORS[model], zorder=4)
        ax.set_title(f"{ep['index']:02d} | {ep.get('short_id', ep['uuid'][:8])}", fontsize=12, pad=9)
        ax.set_xlabel("Executed high-level action index", fontsize=9.5)
        ax.set_ylabel("Height above target (m)", fontsize=10)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.tick_params(labelsize=9)
        style_axis(ax)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=LABELS[m]) for m in COLORS]
    handles.append(Line2D([], [], color="#5A626A", lw=1, linestyle="--", label="Target altitude"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.895),
               ncol=3, frameon=False, fontsize=12)
    fig.text(0.05, 0.063,
             "One point per action boundary, sampled from saved positions[::5]. "
             "Positive values are above the target; negative values are below it.",
             fontsize=10.5, color="#545D66")
    fig.text(0.05, 0.035,
             "Baseline action count = (saved states - 1) / 5. "
             "Historical results are matched by UUID; equal action indices do not imply equal elapsed time.",
             fontsize=10.5, color="#545D66")
    save_figure(fig, "altitude_comparison", out, plt)


def main():
    args = parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
    font_cache_dir = out / ".cache"
    font_cache_dir.mkdir(exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(font_cache_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#69727B",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    data = json.loads(args.data.read_text(encoding="utf-8"))
    episodes = validate(data)
    overview(data, episodes, out, plt, np)
    trajectories(episodes, out, plt, np, Line2D)
    altitude(episodes, out, plt, np, Line2D, MaxNLocator)


if __name__ == "__main__":
    main()
