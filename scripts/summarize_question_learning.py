"""Compare question forms, input controls, language retention and data exposure."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def normalized_answer(text):
    value = text.strip().casefold()
    return value[:-1].rstrip() if value.endswith((".", "。")) else value


def averages(rows):
    values = defaultdict(list)
    for row in rows:
        for key, value in row["metrics"].items():
            values[key].append(value)
    return {key: {"mean": sum(samples) / len(samples), "count": len(samples)}
            for key, samples in values.items()}


def constant_answer_baselines(rows):
    counts = defaultdict(Counter)
    for row in rows:
        for response in (row.get("response") or {}).get("responses", []):
            if "expected" in response:
                counts[f"{row['form']}/{row['wording']}"][response["expected"].strip()] += 1
    return {form: {"answers": sum(values.values()), "answer_counts": dict(values),
                   "best_constant_accuracy_on_panel": max(values.values()) / sum(values.values())}
            for form, values in counts.items()}


def paired_answers(rows):
    result = {}
    indexed = {(row["record_key"], row["form"], row["wording"]): row for row in rows}
    for first, second in (("same", "different"), ("legal", "illegal"), ("verify_yes", "verify_no")):
        for wording in (0, 1):
            pairs = [(row, indexed[(key, second, wording)])
                     for (key, form, variant), row in indexed.items()
                     if form == first and variant == wording]
            if not pairs:
                continue
            correct, normalized, changed = [], [], []
            for left, right in pairs:
                a, b = left["response"]["responses"][0], right["response"]["responses"][0]
                if {a["expected"], b["expected"]} != {"yes", "no"}:
                    raise ValueError("paired questions do not have complementary answers")
                if "answer" in a and "answer" in b:
                    correct.append(a["answer"].strip() == a["expected"] and b["answer"].strip() == b["expected"])
                    normalized.append(normalized_answer(a["answer"]) == a["expected"] and normalized_answer(b["answer"]) == b["expected"])
                    changed.append(normalized_answer(a["answer"]) != normalized_answer(b["answer"]))
            result[f"{first}+{second}/{wording}"] = {
                "pairs": len(pairs), "generated_pairs": len(correct),
                "both_correct": sum(correct), "both_correct_rate": sum(correct) / len(correct) if correct else None,
                "normalized_both_correct": sum(normalized),
                "normalized_both_correct_rate": sum(normalized) / len(normalized) if normalized else None,
                "different_normalized_answers": sum(changed),
                "teacher_forced_both_correct": sum(a["metrics"]["teacher_forced_exact"] == 1
                                                   and b["metrics"]["teacher_forced_exact"] == 1 for a, b in pairs),
                "constant_relation_baselines": {
                    "first_yes_second_no": sum(a["response"]["responses"][0]["expected"] == "yes" for a, _ in pairs) / len(pairs),
                    "first_no_second_yes": sum(a["response"]["responses"][0]["expected"] == "no" for a, _ in pairs) / len(pairs),
                },
            }
    return result


def language_scores(report):
    result = {}
    for name, rows in (("all", report["generations"]), ("original", report["generations"][:12]),
                       ("additional", report["generations"][12:])):
        scored = [row for row in rows if "expected" in row]
        result[name] = {"correct": sum(row["exact_match"] for row in scored), "questions": len(scored)}
        result[name]["by_language"] = {}
        for language in ("en", "ja"):
            selected = [row for row in scored
                        if ("ja" if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", row["prompt"]) else "en") == language]
            result[name]["by_language"][language] = {"correct": sum(row["exact_match"] for row in selected), "questions": len(selected)}
    result["responses"] = report["generations"]
    scored = [row for row in report["generations"] if "expected" in row]
    arithmetic = [row for row in scored if re.fullmatch(r"-?\d+", row["expected"])]
    matching_numerals = [row for row in arithmetic
                        if (numerals := re.findall(r"-?\d+", row["answer"]))
                        and numerals[-1] == row["expected"]]
    result["format_diagnostics"] = {
        "yes_only_answers": sum(normalized_answer(row["answer"]) == "yes" for row in scored),
        "arithmetic_questions": len(arithmetic),
        "arithmetic_strict_correct": sum(row["exact_match"] for row in arithmetic),
        "arithmetic_final_numeral_matches": len(matching_numerals),
        "scope": "Post-hoc format diagnostic: the final signed integer matches the expected number. This is not a semantic-equivalence score and does not replace strict instruction scoring.",
    }
    return result


def paired_selection_answers(rows):
    indexed = {(row["record_key"], row["form"], row["wording"]): row for row in rows}
    output = {}
    for first, last in (("first_word", "last_word"), ("first_action", "last_action")):
        for wording in (0, 1):
            pairs = [(row, indexed[(key, last, wording)])
                     for (key, form, variant), row in indexed.items()
                     if form == first and variant == wording]
            if not pairs:
                continue
            targets = [(left["response"]["responses"][0]["expected"], right["response"]["responses"][0]["expected"])
                       for left, right in pairs]
            item = {}
            for cohort in ("all", "different_targets"):
                selected = [(pair, target) for pair, target in zip(pairs, targets)
                            if cohort == "all" or target[0] != target[1]]
                generated = [pair for pair, _ in selected if all("exact_match" in row["metrics"] for row in pair)]
                correct = sum(all(row["metrics"]["exact_match"] == 1 for row in pair) for pair in generated)
                counts = Counter(target for _, target in selected)
                item[cohort] = {
                    "pairs": len(selected), "generated_pairs": len(generated),
                    "both_correct": correct,
                    "both_correct_rate": correct / len(generated) if generated else None,
                    "best_constant_answer_pair_accuracy_on_panel": max(counts.values()) / len(selected) if selected else None,
                }
            output[f"{first}+{last}/{wording}"] = item
    return output


def summarize_report(report, training_passages, consumed_passages=None):
    report = copy.deepcopy(report)
    # Keep punctuation/case exact for explicit text copying and the 90 strict
    # language probes. Class/yes-no answers also get a narrowly normalized score.
    for container in (report["sources"], report.get("question_without_observations", {})):
        for name, source in container.items():
            if name in ("tinystories", "wikitext2", "shakespeare", "conversations"):
                continue
            for row in source["rows"]:
                responses = (row.get("response") or {}).get("responses", [])
                responses = [answer for answer in responses if "answer" in answer and "expected" in answer]
                if responses:
                    row["metrics"]["normalized_match"] = sum(normalized_answer(answer["answer"]) == normalized_answer(answer["expected"])
                                                              for answer in responses) / len(responses)
    output = {"step": report["step"], "language": language_scores(report), "sources": {}}
    for name, source in report["sources"].items():
        rows = source["rows"]
        forms = {f"{form}/{wording}": averages([row for row in rows if row["form"] == form and row["wording"] == wording])
                 for form, wording in sorted({(row["form"], row["wording"]) for row in rows})}
        item = {"original": averages([row for row in rows if row["form"] == "original"]),
                "forms": forms, "paired_answers": paired_answers(rows),
                "paired_selection_answers": paired_selection_answers(rows),
                "constant_answer_baselines": constant_answer_baselines(rows)}
        if name == "boolq":
            cohorts = [("passage_cohorts", training_passages, ("new_passage", "passage_in_training_population"))]
            if consumed_passages is not None:
                cohorts.append(("consumption_cohorts", consumed_passages, ("not_read_during_training", "read_during_training")))
            for field, known, labels in cohorts:
                item[field] = {}
                for seen in (False, True):
                    selected = [row for row in rows if (row["group"] in known) == seen]
                    item[field][labels[seen]] = {
                        "distinct_passages": len({row["group"] for row in selected}),
                        "original": averages([row for row in selected if row["form"] == "original"]),
                        "paired_answers": paired_answers(selected),
                    }
        omitted = report.get("question_without_observations", {}).get(name, {}).get("rows", [])
        if omitted:
            complete = {row["key"]: row for row in rows}
            item["input_controls"] = {}
            item["paired_input_controls"] = {
                "complete": paired_answers([complete[row["key"]] for row in omitted]),
                "without_observations": paired_answers(omitted),
            }
            item["selection_input_controls"] = {
                "complete": paired_selection_answers([complete[row["key"]] for row in omitted]),
                "without_observations": paired_selection_answers(omitted),
            }
            for form in sorted({row["form"] for row in omitted}):
                without = [row for row in omitted if row["form"] == form]
                with_inputs = [complete[row["key"]] for row in without]
                item["input_controls"][form] = {"complete": averages(with_inputs), "without_observations": averages(without)}
        if name == "native_experience":
            item["native_input_controls"] = {
                condition: averages([row for row in control[name]["rows"] if row["form"] == "original"])
                for condition, control in report.get("native_input_controls", {}).items()
            }
        output["sources"][name] = item
    return output


def native_baselines(root, data_root, panel):
    import torch

    from intrep.experience.multimodal.records import episode_digest, load_episode
    from intrep.problems.shared_prediction.streams import file_identity

    torch.set_num_threads(2)
    config = next(row for row in read_json(root / "recipe.json")["sources"] if row["kind"] == "native")
    selection_path = data_root / config["selection"]
    expected = read_json(root / "provenance.json")["evaluation"][config["name"]]["reader"]["selection"]
    if file_identity(selection_path) != expected:
        raise ValueError("native baseline selection changed")
    entries = read_json(selection_path)["episodes"]
    training = Counter()
    for entry in entries:
        if entry["split"] == "train":
            payload = read_json(selection_path.parent / entry["path"])
            teachers = payload.get("targets", {}).get("teacher_actions", [])
            for index, action in enumerate(payload["actions"]):
                training[teachers[index] if teachers and teachers[index] is not None else action] += 1
    majority = min(training, key=lambda action: (-training[action], action))
    validation = [entry for entry in entries if entry["split"] == config["evaluation"]["split"]]
    cached, targets, metrics = {}, {}, []
    for case in panel[config["name"]]:
        if case["form"] != "original":
            continue
        index = case["episode"]
        if index not in cached:
            entry = validation[index]
            path = selection_path.parent / entry["path"]
            if episode_digest(path) != entry["sha256"]:
                raise ValueError("native baseline episode changed")
            cached[index] = load_episode(path)
        episode, transition = cached[index], case["transition"]
        teacher = episode.teacher_actions[transition] if episode.teacher_actions else None
        targets[case["key"]] = episode.actions[transition] if teacher is None else teacher
        observed, future = episode.observations[transition:transition + 2]
        metrics.append({"metrics": {
            "image_persistence_mse": float((future.image - observed.image).square().mean()),
            "audio_persistence_mse": float((future.audio - observed.audio).square().mean()),
            "silent_audio_mse": float(future.audio.square().mean()),
        }})
    return {"training_action_counts": dict(training), "majority_action": majority,
            "majority_accuracy": sum(action == majority for action in targets.values()) / len(targets),
            "validation_action_counts": dict(Counter(targets.values())), "forecasts": averages(metrics)}, targets


def completion_baselines(root, data_root, report, panel):
    import torch

    from intrep.problems.shared_prediction.questions import (
        image_question,
        sensor_question,
        waveform_question,
    )
    from intrep.problems.shared_prediction.recipe import evaluation_recipe
    from intrep.problems.shared_prediction.record_sources import (
        SensorSource,
        SpokenSource,
    )
    from intrep.problems.shared_prediction.sources import (
        ClassificationSource,
        source_configs,
    )

    # Readers only need a parameter to choose their device/dtype; no model is run.
    device_reference = torch.nn.Linear(1, 1)
    configurations = source_configs(evaluation_recipe(read_json(root / "recipe.json")))
    factories = {"idx": ClassificationSource, "cifar10": ClassificationSource,
                 "spoken_digits": SpokenSource, "inertial_activity": SensorSource}
    output = {}
    for config in configurations:
        if config["kind"] not in factories:
            continue
        name, rows = config["name"], []
        reader = factories[config["kind"]](device_reference, None, config, data_root)
        if reader.provenance() != read_json(root / "provenance.json")["evaluation"][name]["reader"]:
            raise ValueError("completion baseline source differs from evaluation data")
        cases = {case["key"]: case for case in panel[name]}
        for row in report["sources"][name]["rows"]:
            form = row["form"]
            if row["wording"] != 0 or form not in ("inpaint", "audio_gap", "sensor_future"):
                continue
            losses, existing = [], []
            for index in row["response"]["record_indices"]:
                record = reader.read_record(index)
                if form == "inpaint":
                    question = image_question(reader, record, cases[row["key"]]["seed"], 0)
                elif form == "audio_gap":
                    question = waveform_question(reader, record, 0)
                else:
                    question = sensor_question(record, 0)
                prediction = question.predictions[0]
                valid = prediction.valid if prediction.valid is not None else torch.ones_like(prediction.target, dtype=torch.bool)
                target = prediction.target[valid]
                losses.append(float(target.square().mean()))
                existing.append(float((prediction.baseline[valid] - target).square().mean()))
            baseline = sum(existing) / len(existing)
            saved = row["metrics"][f"{prediction.head}_baseline_mse"]
            if abs(saved - baseline) > 1e-5 * max(1, abs(saved)):
                raise ValueError("reconstructed completion mask/target differs from the saved baseline")
            rows.append({"metrics": {"zero_mse": sum(losses) / len(losses), "existing_baseline_mse": baseline}})
        output[name] = averages(rows)
    return output


def make_plots(root, conditions, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(4, 3, figsize=(12, 12), constrained_layout=True)
    for name, axis in zip(conditions["fixed"]["after"]["sources"], axes.flat):
        for mode, color in (("fixed", "#28699a"), ("varied", "#b45531")):
            records = curves[mode]
            axis.plot([record["step"] for record in records],
                      [record["sources"][name]["forms"]["original/0"]["loss"]["mean"] for record in records],
                      marker="o", color=color, label=mode)
        axis.set(title=name.replace("_", " "), xlabel="Joint updates", ylabel="Original objective loss")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Original objectives · same development examples · one-hour training budgets", fontsize=13)
    figure.savefig(root / "original-learning-curves.png", dpi=160)
    figure.savefig(root / "original-learning-curves.pdf")
    plt.close(figure)

    names, fixed, varied, baselines = [], [], [], []
    for name, source in conditions["fixed"]["after"]["sources"].items():
        for pair, row in source["paired_answers"].items():
            if pair.endswith("/1"):
                names.append(name.replace("_", " "))
                fixed.append(row["normalized_both_correct_rate"])
                varied.append(conditions["varied"]["after"]["sources"][name]["paired_answers"][pair]["normalized_both_correct_rate"])
                baselines.append(max(row["constant_relation_baselines"].values()))
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    positions = list(range(len(names)))
    axis.barh([i + 0.18 for i in positions], fixed, height=0.34, color="#28699a", label="fixed")
    axis.barh([i - 0.18 for i in positions], varied, height=0.34, color="#b45531", label="varied")
    for index, baseline in enumerate(baselines):
        axis.vlines(baseline, index - 0.4, index + 0.4, color="#555555", linestyle="--",
                    label="Best constant relation on this panel" if index == 0 else None)
    axis.set(yticks=positions, yticklabels=names, xlim=(0, 1), xlabel="Both answers correct (case / final period normalized)",
             title="Complementary yes/no questions · held-out wording · greedy generation")
    axis.legend(frameon=False)
    figure.savefig(root / "question-pairs.png", dpi=160)
    figure.savefig(root / "question-pairs.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    args = parser.parse_args()
    train = args.data_root / "data/question-learning-20260911/boolq/train.jsonl"
    passage_sequence = [hashlib.sha256(json.loads(line)["passage"].encode()).hexdigest() for line in train.read_text().splitlines()]
    passages = set(passage_sequence)
    conditions, curves, panel, initial_hash = {}, {}, None, None
    for mode in ("fixed", "varied"):
        root = args.root / mode
        result = read_json(root / "result.json")
        current_panel = read_json(root / "evaluation-panel.json")
        if panel is not None and (panel != current_panel or initial_hash != result["initial_parameters_sha256"]):
            raise ValueError("comparison conditions do not share their panel and initial parameters")
        panel, initial_hash = current_panel, result["initial_parameters_sha256"]
        reports = [read_json(path) for path in sorted((root / "evaluation").glob("step-*.json"))]
        curves[mode] = reports
        # BoolQ traverses complete records in file order, one per update. A
        # passage in the available population has not necessarily been read.
        boolq_records = result["source_progress"]["boolq"]["reader"]["records"]
        if boolq_records != result["completed_steps"]:
            raise ValueError("BoolQ exposure differs from its one-record-per-update schedule")
        consumed_passages = set(passage_sequence[:boolq_records])
        item = {"before": summarize_report(reports[0], passages, consumed_passages), "after": summarize_report(reports[-1], passages, consumed_passages),
                "completed_steps": result["completed_steps"], "training_seconds": result["training_seconds"],
                "trainable_parameters": result["trainable_parameters"], "cuda_peak_allocated_mib": result["cuda_peak_allocated_mib"],
                "source_progress": result["source_progress"], "consumed_records": {
                    name: sum(step["source_details"][name]["records"] for step in result["joint_updates"])
                    for name in result["source_progress"]},
                "native_baselines": None}
        item["native_baselines"], targets = native_baselines(root, args.data_root, panel)
        item["completion_baselines"] = completion_baselines(root, args.data_root, reports[0], panel)
        for stage, report in (("before", reports[0]), ("after", reports[-1])):
            rows = [row for row in report["sources"]["native_experience"]["rows"] if row["form"] == "original"]
            per_action = {str(action): averages([row for row in rows if targets[row["key"]] == action])["action_accuracy"]["mean"]
                          for action in sorted(set(targets.values()))}
            item[stage]["sources"]["native_experience"]["balanced_action_accuracy"] = {
                "per_action": per_action, "macro_mean": sum(per_action.values()) / len(per_action)}
        conditions[mode] = item
    common_steps = sorted({row["step"] for row in curves["fixed"]} & {row["step"] for row in curves["varied"]})
    output = {"conditions": conditions, "common_evaluation_steps": common_steps,
              "initial_parameters_sha256": initial_hash,
              "limitations": "One seed; development panels; paraphrases of trained tasks; training wall time is not exact FLOPs. Input omission also changes the sequence distribution."}
    (args.root / "comparison.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    make_plots(args.root, conditions, curves)
    print(json.dumps({mode: {"steps": item["completed_steps"], "training_seconds": item["training_seconds"],
                            "language": {stage: item[stage]["language"]["all"] for stage in ("before", "after")}}
                      for mode, item in conditions.items()}))


if __name__ == "__main__":
    main()
