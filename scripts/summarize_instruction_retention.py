"""Summarize the prespecified three-condition instruction retention pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.summarize_question_learning import (
    completion_baselines,
    native_baselines,
    read_json,
    summarize_report,
)

CONDITIONS = ("chunked", "assistant", "assistant_weighted")


def scored_prompts(report, split):
    selected = {}
    for row in report["generations"]:
        if row["split"] != split or "expected" not in row:
            continue
        if row["prompt"] in selected:
            raise ValueError("instruction panel contains duplicate prompts")
        if row["exact_match"] != (row["answer"].strip() == row["expected"]):
            raise ValueError("stored exact-match score does not match the generated answer")
        selected[row["prompt"]] = row
    return selected


def instruction_scores(report, split, before=None):
    selected = scored_prompts(report, split)
    rows = list(selected.values())
    groups = defaultdict(list)
    for row in rows:
        language = row.get("language") or ("ja" if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", row["prompt"]) else "en")
        groups[f"language/{language}"].append(row)
        if "family" in row:
            groups[f"family/{row['family']}"].append(row)

    def counts(cases):
        return {"correct": sum(bool(row["exact_match"]) for row in cases), "questions": len(cases)}

    result = {**counts(rows), "groups": {group: counts(cases) for group, cases in sorted(groups.items())}}
    arithmetic = [row for row in rows if re.fullmatch(r"-?\d+", row["expected"])]
    matching = [row for row in arithmetic if (numbers := re.findall(r"-?\d+", row["answer"])) and numbers[-1] == row["expected"]]
    result["format_diagnostic"] = {
        "numeric_questions": len(arithmetic), "strict_correct": sum(row["exact_match"] for row in arithmetic),
        "final_numeral_matches": len(matching), "scope": "Last-numeral diagnostic only; not semantic equivalence.",
    }
    if before is not None:
        original = scored_prompts(before, split)
        if selected.keys() != original.keys() or any(selected[key]["expected"] != original[key]["expected"] for key in selected):
            raise ValueError("instruction retention requires identical prompts and targets")
        result["paired"] = {
            "initially_correct": sum(row["exact_match"] for row in original.values()),
            "retained_correct": sum(original[key]["exact_match"] and row["exact_match"] for key, row in selected.items()),
            "lost_correct": sum(original[key]["exact_match"] and not row["exact_match"] for key, row in selected.items()),
            "newly_correct": sum(not original[key]["exact_match"] and row["exact_match"] for key, row in selected.items()),
        }
    return result


def gradient_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        probe = record.get("gradient_probe")
        if not probe:
            continue
        form = record["source_details"]["conversations"]["form"]
        cohort = "conversation_original" if form == "original" else "conversation_excerpt_question"
        for source, values in probe["sources"].items():
            for metric, value in values.items():
                if value is not None:
                    grouped[f"{cohort}/{source}"][metric].append(value)
    return {key: {metric: {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "samples": len(values)}
                  for metric, values in metrics.items()} for key, metrics in sorted(grouped.items())}


def summarize_condition(directory, training_passages):
    result = read_json(directory / "result.json")
    initial = read_json(directory / "evaluation/step-000000.json")
    final = read_json(directory / "evaluation" / f"step-{result['completed_steps']:06d}.json")
    generations = [read_json(path) for path in sorted((directory / "generations").glob("step-*.json"))]
    if generations[0]["step"] != 0 or generations[-1]["step"] != result["completed_steps"]:
        raise ValueError("instruction measurement does not cover initial and final models")
    for report in generations[1:-1]:
        if any(row["split"] == "holdout" for row in report["generations"]):
            raise ValueError("fresh held-out prompts were evaluated during intermediate model selection")
    records = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
    archive = read_json(directory / "archive.json")
    verification = read_json(directory / "cpu-verification.json")
    if not archive["verified"] or not verification["parameter_digest_matches_training"]:
        raise ValueError("the condition lacks verified model storage")
    if verification["final_parameters_sha256"] != result["final_parameters_sha256"]:
        raise ValueError("archived checkpoint differs from final evaluation")
    return {
        "completed_steps": result["completed_steps"], "training_seconds": result["training_seconds"],
        "parameters": result["parameters"], "trainable_parameters": result["trainable_parameters"],
        "initial_parameters_sha256": result["initial_parameters_sha256"],
        "final_parameters_sha256": result["final_parameters_sha256"],
        "core_parameter_tensors": result["core_parameter_tensors"],
        "changed_core_parameter_tensors": result["changed_core_parameter_tensors"],
        "cuda_peak_allocated_mib": result["cuda_peak_allocated_mib"],
        "source_progress": result["source_progress"],
        "development_curve": [{"step": report["step"], **instruction_scores(report, "development", initial)} for report in generations],
        "holdout_before": instruction_scores(initial, "holdout"),
        "holdout_after": instruction_scores(final, "holdout", initial),
        "sources_before": summarize_report(initial, training_passages)["sources"],
        "sources_after": summarize_report(final, training_passages)["sources"],
        "gradient_summary": gradient_summary(records),
        "archive": archive, "cpu_verification": verification,
    }, initial, final


def plot_results(root, conditions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"chunked": "128-token all-role", "assistant": "Assistant only", "assistant_weighted": "Assistant only, weight 8"}
    colors = {"chunked": "#55585e", "assistant": "#267694", "assistant_weighted": "#b05621"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    for name, condition in conditions.items():
        curve = condition["development_curve"]
        axes[0].plot([row["step"] for row in curve], [100 * row["correct"] / row["questions"] for row in curve],
                     marker="o", color=colors[name], label=names[name], linewidth=2, markersize=4)
    axes[0].set(title="Instruction retention during training", xlabel="Joint optimizer updates", ylabel="Development exact match (%)", ylim=(0, 100))
    axes[0].grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    baseline = next(iter(conditions.values()))["holdout_before"]
    labels = ["Initial", *[names[name] for name in conditions]]
    values = [100 * baseline["correct"] / baseline["questions"]]
    values.extend(100 * row["holdout_after"]["correct"] / row["holdout_after"]["questions"] for row in conditions.values())
    bars = axes[1].bar(range(4), values, color=["#babcc0", *colors.values()], width=.65)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    axes[1].set(title="Fresh bilingual prompts: final models", ylabel="Held-out exact match (%)", ylim=(0, 100))
    axes[1].set_xticks(range(4), ["Initial", "All-role", "Assistant", "Assistant\nweight 8"])
    axes[1].grid(axis="y", alpha=.15)
    axes[1].set_axisbelow(True)
    figure.suptitle("One shared LFM2.5-350M · all parameters trained · 300 updates per condition", fontsize=12)
    figure.savefig(root / "instruction-retention.png", dpi=200)
    figure.savefig(root / "instruction-retention.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    args = parser.parse_args()
    training_passages = {hashlib.sha256(json.loads(line)["passage"].encode()).hexdigest()
                        for line in (args.data_root / "data/question-learning-20260911/boolq/train.jsonl").open()}
    conditions, initials, finals = {}, {}, {}
    for name in CONDITIONS:
        conditions[name], initials[name], finals[name] = summarize_condition(args.root / name, training_passages)
    if len({row["initial_parameters_sha256"] for row in conditions.values()}) != 1:
        raise ValueError("initial parameters differ between conditions")
    if len({row["completed_steps"] for row in conditions.values()}) != 1:
        raise ValueError("conditions have different optimizer update budgets")
    panels = [read_json(args.root / name / "evaluation-panel.json") for name in CONDITIONS]
    if not panels[0] == panels[1] == panels[2]:
        raise ValueError("validation panel locations differ")
    for name in CONDITIONS[1:]:
        if initials[name]["generations"] != initials[CONDITIONS[0]]["generations"]:
            raise ValueError("initial instruction responses differ")
        for source, rows in initials[CONDITIONS[0]]["sources"].items():
            if source != "conversations" and initials[name]["sources"][source] != rows:
                raise ValueError(f"initial non-conversation evaluation differs: {source}")
    native, _ = native_baselines(args.root / CONDITIONS[0], args.data_root, panels[0])
    completions = completion_baselines(args.root / CONDITIONS[0], args.data_root, finals[CONDITIONS[0]], panels[0])
    summary = {
        "schema_version": "intrep.instruction-retention-comparison.v1",
        "conditions": conditions, "native_baselines": native, "completion_baselines": completions,
        "controls_verified": {"initial_parameters": True, "initial_generations": True,
                              "initial_non_conversation_evaluations": True, "panel_locations": True},
        "limitations": [
            "One training seed, small fixed panels, and a short pilot; no long-term retention conclusion.",
            "The first contrast changes conversation context and labels together; consumed records and tokens differ.",
            "Conversation excerpts also differ between chunked and assistant readers; compare them only within matching readers.",
            "The second contrast changes only conversation source weight; weights are not whole-model gradient shares.",
            "Fresh prompts are exactly separate from training user turns, but semantic overlap and pretrained exposure are not excluded.",
            "All-role controls use the same expanded bilingual corpus; this is not an exact repeat of the previous run.",
        ],
    }
    (args.root / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    plot_results(args.root, conditions)
    print(json.dumps({"conditions": {name: {"development": row["development_curve"][-1]["correct"],
                                           "holdout": row["holdout_after"]["correct"]}
                                      for name, row in conditions.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
