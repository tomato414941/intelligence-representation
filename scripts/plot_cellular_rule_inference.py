"""Render a standalone PNG/PDF and CSV from unseen-rule evaluation JSON.

Run with: uv run --no-project --with matplotlib python scripts/plot_cellular_rule_inference.py ...
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text())
    rows = data["summaries"]
    counts = [r["context_count"] for r in rows]
    positions = np.arange(len(counts))
    cfg = data["training_config"]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for axis in axes:
        axis.set_xticks(positions, counts)
        axis.set_xlabel("Observed before/after examples")
        axis.set_ylim(0, 1.03)
        axis.set_yticks(np.linspace(0, 1, 6), ["0%", "20%", "40%", "60%", "80%", "100%"])
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.8)
        axis.set_axisbelow(True)
    ax = axes[0]
    for name, label, color, style in [
        ("correct_context", "Correct-rule examples", "#1261a0", "-"),
        ("wrong_context", "Other-rule examples", "#bf4e5d", "-"),
        ("frequency_baseline", "Frequency baseline (no geometry)", "#64748b", "--"),
        ("rule_family_lookup", "Lookup baseline (given rule family)", "#44846b", ":"),
    ]:
        values = [row[name]["accuracy"] for row in rows]
        ax.plot(positions, values, style, color=color, marker="o", label=label, linewidth=2)
    intervals = np.array([row["correct_context"]["rule_bootstrap_ci95"] for row in rows])
    ax.fill_between(positions, intervals[:, 0], intervals[:, 1], alpha=0.14, color="#1261a0")
    ax.set_title("Prediction on held-out rules", loc="left", pad=12)
    ax.set_ylabel("Query-cell accuracy")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax = axes[1]
    for name, label, color, style in [
        ("covered_accuracy", "Conditions observed in examples", "#1261a0", "-"),
        ("uncovered_accuracy", "Conditions absent from examples", "#bf4e5d", "-"),
        ("coverage", "Fraction of query cells covered", "#64748b", "--"),
    ]:
        values = [row["correct_context"][name] for row in rows]
        ax.plot(positions, [np.nan if value is None else value for value in values], style,
                color=color, marker="o", linewidth=2, label=label)
    ax.set_title("What the examples make identifiable", loc="left", pad=12)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("Can observations reveal an unseen cellular rule?", x=0.07, ha="left", fontsize=17, weight="bold")
    subtitle = (f"{cfg['model']['height']}×{cfg['model']['width']} grids  ·  {len(data['eval_rules'])} unseen rules × "
                f"{data['queries_per_rule']} query boards  ·  {data['training_step']:,} training steps  ·  model seed {cfg['model_seed']}")
    fig.text(0.07, 0.905, subtitle, color="#475569", fontsize=10)
    fig.text(0.07, 0.025, "Fixed weights at evaluation. Shading: 95% bootstrap interval across rules.\n"
             "Coverage and lookup use rule-family knowledge for analysis only; the neural model receives raw cell pairs.",
             color="#475569", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.17, top=0.8, wspace=0.23)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=180)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["context_count", "accuracy", "ci95_low", "ci95_high", "wrong_context_accuracy",
                         "frequency_baseline", "rule_lookup", "covered_accuracy", "uncovered_accuracy", "coverage"])
        for row in rows:
            correct = row["correct_context"]
            writer.writerow([row["context_count"], correct["accuracy"], *correct["rule_bootstrap_ci95"],
                             row["wrong_context"]["accuracy"], row["frequency_baseline"]["accuracy"],
                             row["rule_family_lookup"]["accuracy"], correct["covered_accuracy"],
                             correct["uncovered_accuracy"], correct["coverage"]])
    print(args.output_prefix.with_suffix(".png"))


if __name__ == "__main__":
    main()
