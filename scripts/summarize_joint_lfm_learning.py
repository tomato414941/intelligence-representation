"""Summarize paired development results and compare native forecasts to persistence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from intrep.experience.multimodal.records import episode_digest, load_episode
from intrep.problems.shared_prediction.streams import file_identity


def read_json(path):
    return json.loads(path.read_text())


def native_baselines(root, data_root, recipe, panel):
    config = next(row for row in recipe["sources"] if row["kind"] == "native")
    name = config["name"]
    selection_path = data_root / config["selection"]
    expected = read_json(root / "provenance.json")["evaluation"][name]["selection"]
    if file_identity(selection_path) != expected:
        raise ValueError("native baseline data differs from the evaluation provenance")
    entries = read_json(selection_path)["episodes"]
    training_counts = Counter()
    for entry in entries:
        if entry["split"] != "train":
            continue
        payload = read_json(selection_path.parent / entry["path"])
        teachers = payload.get("targets", {}).get("teacher_actions", [])
        for index, action in enumerate(payload["actions"]):
            teacher = teachers[index] if teachers else None
            training_counts[action if teacher is None else teacher] += 1
    majority = min(training_counts, key=lambda action: (-training_counts[action], action))
    validation = [entry for entry in entries if entry["split"] == config["evaluation"]["split"]]
    cached = {}
    targets, images, audio, silence = {}, [], [], []
    for case in panel[name]:
        index = case["episode"]
        if index not in cached:
            entry = validation[index]
            path = selection_path.parent / entry["path"]
            if episode_digest(path) != entry["sha256"]:
                raise ValueError("native episode differs from its recorded digest")
            cached[index] = load_episode(path)
        episode = cached[index]
        transition = case["transition"]
        teacher = episode.teacher_actions[transition] if episode.teacher_actions else None
        targets[case["key"]] = episode.actions[transition] if teacher is None else teacher
        observed, future = episode.observations[transition:transition + 2]
        if observed.image is not None and future.image is not None and observed.image.shape == future.image.shape:
            images.append(float((future.image - observed.image).square().mean()))
        if future.audio is not None:
            silence.append(float(future.audio.square().mean()))
            if (observed.audio is not None and observed.audio.shape == future.audio.shape
                    and observed.sample_rate == future.sample_rate):
                audio.append(float((future.audio - observed.audio).square().mean()))
    baselines = {"training_action_counts": dict(training_counts), "training_majority_action": majority,
                 "majority_action_accuracy": sum(action == majority for action in targets.values()) / len(targets),
                 "validation_action_counts": dict(Counter(targets.values())),
                 "image_persistence_mse": float(np.mean(images)) if images else None, "image_examples": len(images),
                 "audio_persistence_mse": float(np.mean(audio)) if audio else None, "audio_examples": len(audio),
                 "silent_audio_mse": float(np.mean(silence)) if silence else None, "worlds": len(cached)}
    return name, baselines, targets


def balanced_accuracy(report, source, targets):
    by_action = {}
    for action in set(targets.values()):
        scores = [row["metrics"]["action_accuracy"] for row in report["sources"][source]["rows"]
                  if targets[row["key"]] == action]
        by_action[str(action)] = float(np.mean(scores))
    return {"per_action": by_action, "macro_mean": float(np.mean(list(by_action.values())))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    args = parser.parse_args()
    torch.set_num_threads(2)
    summaries, curves = {}, {}
    canonical_panel = None
    for size in ("230m", "350m"):
        root = args.root / size
        result = read_json(root / "result.json")
        panel = read_json(root / "evaluation-panel.json")
        if canonical_panel is not None and canonical_panel != panel:
            raise ValueError("model comparisons used different evaluation examples")
        canonical_panel = panel
        evaluations = [read_json(path) for path in sorted((root / "evaluation").glob("step-*.json"))]
        before, after = evaluations[0], evaluations[-1]
        source, baselines, targets = native_baselines(root, args.data_root, read_json(root / "recipe.json"), panel)
        controls = {}
        for stage, report in (("before", before), ("after", after)):
            controls[stage] = {condition: value[source]["summary"]
                               for condition, value in report.get("native_input_controls", {}).items()}
        generation = {}
        for stage, report in (("before", before), ("after", after)):
            rows = [row for row in report["generations"] if "expected" in row]
            generation[stage] = {"correct": sum(row["exact_match"] for row in rows), "questions": len(rows), "all_answers": report["generations"]}
        summaries[size] = {"paired_evaluation": result["paired_evaluation"], "native_baselines": baselines,
                           "balanced_native_accuracy": {"before": balanced_accuracy(before, source, targets),
                                                        "after": balanced_accuracy(after, source, targets)},
                           "native_input_controls": controls, "generation": generation,
                           "training_seconds": sum(row["seconds"] for row in result["joint_updates"]),
                           "cuda_peak_allocated_mib": result["cuda_peak_allocated_mib"],
                           "source_progress": result["source_progress"]}
        curves[size] = evaluations
    (args.root / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for name, axis in zip(canonical_panel, axes.flat):
        for size, color in (("230m", "#2667a6"), ("350m", "#b05a2a")):
            records = curves[size]
            axis.plot([row["step"] for row in records],
                      [row["sources"][name]["summary"]["loss"]["mean"] for row in records],
                      marker="o", color=color, label=size.upper())
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Development loss")
        axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Shared model learning · fixed validation examples", fontsize=14)
    figure.savefig(args.root / "learning-curves.png", dpi=180)
    figure.savefig(args.root / "learning-curves.pdf")
    plt.close(figure)
    print(json.dumps({size: {"training_seconds": row["training_seconds"],
                            "native_balanced_accuracy": row["balanced_native_accuracy"],
                            "generation": {stage: {"correct": value["correct"], "questions": value["questions"]}
                                           for stage, value in row["generation"].items()}}
                      for size, row in summaries.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
