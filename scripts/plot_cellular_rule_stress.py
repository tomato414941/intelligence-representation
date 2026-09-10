"""Plot replicated clean inference and the fixed noise/change follow-up."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"clean-seed31": "#65788c", "clean-seed32": "#9aa7b4", "clean-seed33": "#bac2ca",
          "noise-seed31": "#148776", "change-seed31": "#d16a32"}
LABELS = {"clean-seed31": "Clean training / seed 31", "clean-seed32": "Clean / seed 32",
          "clean-seed33": "Clean / seed 33", "noise-seed31": "Noise training / seed 31",
          "change-seed31": "Noise + change training / seed 31"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    models = {name: json.loads((root / name / "stress-test.json").read_text())
              for name in COLORS if (root / name / "stress-test.json").exists()}
    if "clean-seed31" not in models:
        raise ValueError("need the original clean model's stress results")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.7))
    records = []
    for name, evaluation in models.items():
        color = COLORS[name]
        if name.startswith("clean-"):
            clean = json.loads((root / name / "test.json").read_text())
            rows = clean["summaries"]
            axes[0, 0].plot([r["context_count"] for r in rows],
                            [100 * r["correct_context"]["accuracy"] for r in rows],
                            "o-", color=color, label=LABELS[name], linewidth=2)
            for row in rows:
                records.append({"panel": "replication", "model": name, "x": row["context_count"],
                                "accuracy": row["correct_context"]["accuracy"], "brier": ""})
        noise = [r for r in evaluation["noise"] if r["context_count"] == 8]
        change = [r for r in evaluation["change"] if r["noise_rate"] == 0.1 and r["post_change_count"] > 0]
        for axis, rows, x_key, panel in ((axes[0, 1], noise, "noise_rate", "noise"),
                                         (axes[1, 0], change, "post_change_count", "change")):
            x = [100 * r[x_key] if panel == "noise" else r[x_key] for r in rows]
            y = [100 * r["methods"]["model"]["accuracy"] for r in rows]
            ci = [r["methods"]["model"]["accuracy_ci95"] for r in rows]
            axis.plot(x, y, "o-", color=color, linewidth=2, label=LABELS[name])
            axis.fill_between(x, [100*c[0] for c in ci], [100*c[1] for c in ci], color=color, alpha=0.09)
            for row in rows:
                records.append({"panel": panel, "model": name, "x": row[x_key],
                                "accuracy": row["methods"]["model"]["accuracy"],
                                "brier": row["methods"]["model"]["brier"]})
        axes[1, 1].plot([100 * r["noise_rate"] for r in noise],
                        [r["methods"]["model"]["brier"] for r in noise], "o-", color=color, linewidth=2)
    baseline = models["clean-seed31"]
    noise = [r for r in baseline["noise"] if r["context_count"] == 8]
    change = [r for r in baseline["change"] if r["noise_rate"] == 0.1 and r["post_change_count"] > 0]
    axes[0, 1].plot([100*r["noise_rate"] for r in noise],
                    [100*r["methods"]["family_bayes"]["accuracy"] for r in noise],
                    "--", color="#373b40", label="Family Bayes (privileged)")
    axes[1, 0].plot([r["post_change_count"] for r in change],
                    [100*r["methods"]["family_recent4"]["accuracy"] for r in change],
                    "--", color="#373b40", label="Family Bayes / recent 4")
    axes[1, 1].plot([100*r["noise_rate"] for r in noise],
                    [r["methods"]["family_bayes"]["brier"] for r in noise], "--", color="#373b40")
    titles = ("A  Clean inference replicates", "B  Noisy observations, 8 examples",
              "C  Rule change with 10% observation noise", "D  Probability quality, 8 examples")
    x_labels = ("Demonstration pairs", "Corrupted output cells (%)",
                "New-rule examples in an 8-example window", "Corrupted output cells (%)")
    for index, axis in enumerate(axes.flat):
        axis.set_title(titles[index], loc="left", pad=12)
        axis.set_xlabel(x_labels[index])
        axis.set_ylabel("Accuracy (%)" if index < 3 else "Brier score (lower is better)")
        axis.grid(axis="y", alpha=0.18)
        if index < 3:
            axis.set_ylim(45, 101)
    axes[0, 0].set_xticks([0, 1, 4, 8])
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[0, 1].set_xticks([0, 5, 10, 20])
    lowest_noise_accuracy = min(row["methods"]["model"]["accuracy"]
                                for evaluation in models.values() for row in evaluation["noise"]
                                if row["context_count"] == 8)
    axes[0, 1].set_ylim(max(0, ((100 * lowest_noise_accuracy - 3) // 5) * 5), 101)
    axes[1, 1].set_xticks([0, 5, 10, 20])
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 0].set_xticks([1, 2, 4, 6, 8])
    axes[1, 0].text(0.98, 0.04, "Dashed: family Bayes using recent 4", transform=axes[1, 0].transAxes,
                    ha="right", fontsize=8, color="#555d66")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Evidence can be noisy or stale", fontsize=19, weight="bold", x=0.07, ha="left")
    fig.text(0.07, 0.931, "6 × 6 cellular worlds · frozen weights at evaluation · 6,000 training steps per model", color="#555d66")
    fig.text(0.07, 0.018, f"Stress evaluation: {baseline['rule_pairs']} disjoint rule pairs × "
             f"{baseline['queries_per_rule']} queries. Bands: 95% rule-pair bootstrap CI.\n"
             "Augmented models use one training seed. Panel C omits the unobservable zero-evidence change.",
             fontsize=8, color="#555d66")
    fig.tight_layout(rect=(0.04, 0.13, 0.99, 0.92), h_pad=2.4, w_pad=2.6)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(root / f"followup.{extension}", dpi=180, facecolor="white")
    with (root / "followup.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=("panel", "model", "x", "accuracy", "brier"))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
