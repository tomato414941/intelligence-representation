#!/usr/bin/env python3
"""Render the fixed-seed control comparison and export its plotted values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROUNDS = (0, 1, 4, 8)
COLORS = {"experience": "#24665f", "forgetful": "#6c7376", "wrong_context": "#b16b42",
          "family_bayes": "#59799d", "random": "#b0b3b0"}
LABELS = {"experience": "Observed experience", "forgetful": "Forget every task",
          "wrong_context": "Wrong-world experience", "family_bayes": "Known-family baseline",
          "random": "Random action"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.artifact_dir / f"clean-seed{seed}.json" for seed in (31, 32, 33)]
    results = [json.loads(path.read_text()) for path in paths]
    if len({result["task_sha256"] for result in results}) != 1:
        raise ValueError("all checkpoints must use identical control tasks")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#283634",
                         "text.color": "#283634", "axes.edgecolor": "#c7cecb",
                         "figure.facecolor": "#faf9f5", "axes.facecolor": "#faf9f5"})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    metrics = (("regret_cells", 1, "Regret (matching cells lost; lower is better)"),
               ("optimal_action_rate", 100, "Optimal action chosen (%; higher is better)"))
    for ax, (metric, multiplier, ylabel) in zip(axes, metrics):
        for method, color in COLORS.items():
            values = [results[0]["summaries"][turn]["methods"][method][metric] for turn in ROUNDS]
            means = [value["mean"] * multiplier for value in values]
            style = "-" if method == "experience" else "--" if method != "random" else ":"
            ax.plot(ROUNDS, means, style, color=color, marker="o" if method == "experience" else None,
                    lw=2.4 if method == "experience" else 1.5, label=LABELS[method])
            if method == "experience":
                low = [value["rule_pair_bootstrap_ci95"][0] * multiplier for value in values]
                high = [value["rule_pair_bootstrap_ci95"][1] * multiplier for value in values]
                ax.fill_between(ROUNDS, low, high, color=color, alpha=0.12, linewidth=0)
                for seed, other in zip((32, 33), results[1:]):
                    y = [other["summaries"][turn]["methods"][method][metric]["mean"] * multiplier for turn in ROUNDS]
                    ax.plot(ROUNDS, y, color=color, alpha=0.55, lw=1, marker="x" if seed == 32 else "+",
                            label=f"Observed experience, seed {seed}")
        ax.set(xlabel="Previously executed tasks", ylabel=ylabel, xticks=ROUNDS)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 103)
    fig.suptitle("Does executed experience improve the next decision?", fontsize=17, x=0.08, ha="left", y=0.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=9)
    fig.text(0.08, 0.895, "64 unseen worlds × 4 trials · frozen Transformers · 37 one-step interventions", fontsize=10, color="#66706c")
    fig.text(0.08, -0.01, "Main lines: seed 31. Band: 95% world-pair bootstrap. All-tie tasks excluded. Family baseline knows the local rule family.",
             fontsize=8, color="#66706c")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.25, wspace=0.3)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(args.artifact_dir / f"control.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    with (args.artifact_dir / "control.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model_seed", "experience_count", "method", "metric", "mean", "ci95_low", "ci95_high"))
        for seed, result in zip((31, 32, 33), results):
            for turn in ROUNDS:
                for method, data in result["summaries"][turn]["methods"].items():
                    for metric, value in data.items():
                        interval = value["rule_pair_bootstrap_ci95"] or (None, None)
                        writer.writerow((seed, turn, method, metric, value["mean"], *interval))


if __name__ == "__main__":
    main()
