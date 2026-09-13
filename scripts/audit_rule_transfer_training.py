"""Verify that the three rule-transfer conditions isolate their text tuition."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from intrep.problems.shared_prediction.rule_transfer_data import text_training_examples
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_NAMES


def audit_conditions(common, results, traces, common_trace):
    if (set(results) != {"a", "b", "control"} or set(traces) != set(results)
            or common["condition"] != "calibration" or not common["prerequisites_passed"]):
        raise ValueError("compare all three conditions from a passing common calibration")
    steps = {result["completed_steps"] for result in results.values()}
    if len(steps) != 1 or next(iter(steps)) < 1:
        raise ValueError("conditions must receive the same positive number of updates")
    count = next(iter(steps))
    source_names = set(common["source_progress"])
    if len(source_names) != 12:
        raise ValueError("the common checkpoint must include all twelve background sources")
    if [row["step"] for row in common_trace] != list(range(1, common["completed_steps"] + 1)):
        raise ValueError("the common trace must cover every calibration update")
    final_common_state = common_trace[-1]["background_state_sha256"] if common_trace else common["initial_background_state_sha256"]
    comparable = lambda result: {key: value for key, value in result["settings"].items()
                                 if key not in ("condition", "manifest_sha256")}
    reference = comparable(results["a"])
    tuition_count = count * reference["batches"][3]
    if tuition_count % 90:
        raise ValueError("matched tuition must cover complete balanced 90-example cycles")
    for name, result in results.items():
        if (result["condition"] != name or result["settings"]["condition"] != name
                or result["initial_checkpoint_sha256"] != common["checkpoint_sha256"]
                or result["initial_parameters_sha256"] != common["final_parameters_sha256"]
                or result["initial_background_state_sha256"] != final_common_state
                or comparable(result) != reference):
            raise ValueError("branches must share initial weights, readers, panel and training settings")
        if (not result["optimizer_reset_at_fork"] or result["parameters"] != result["trainable_parameters"]
                or result["parameters"] != common["parameters"] or set(result["source_progress"]) != source_names
                or result["new_rule_image_training_examples"] != 0
                or result["new_rule_image_evaluation_queries"] != 0):
            raise ValueError("a branch violated the full-parameter, full-source or modality boundary")
        manifest = result["lessons"]["manifest"]
        expected = text_training_examples(common["lessons"]["orders"], name)
        if (manifest["condition"] != name or manifest["examples"] != expected
                or result["lessons"]["orders"] != common["lessons"]["orders"]):
            raise ValueError("the branch's text tuition differs from the prepared intervention")
        if [row["step"] for row in traces[name]] != list(range(1, count + 1)):
            raise ValueError("the branch trace must cover exactly its declared updates")
        for row in traces[name]:
            if (set(row["losses"]) != source_names | set(LESSON_NAMES) | {"weighted_loss", "grad_norm"}
                    or any(not math.isfinite(value) for value in row["losses"].values())
                    or set(row["lesson_inputs"]) != set(LESSON_NAMES)
                    or any(len(row["lesson_inputs"][lesson]) != reference["batches"][index]
                           for index, lesson in enumerate(LESSON_NAMES))):
                raise ValueError("every update must contain finite losses for every source and lesson")
        seen = Counter(identifier for row in traces[name] for identifier in row["lesson_inputs"]["text_tuition"])
        if seen != Counter({row["id"]: tuition_count // 90 for row in expected}):
            raise ValueError("every textual pair must receive the same tuition exposure")
    for rows in zip(*(traces[name] for name in ("a", "b", "control")), strict=True):
        if (len({row["background_state_sha256"] for row in rows}) != 1
                or any(row["lesson_inputs"] != rows[0]["lesson_inputs"] for row in rows[1:])):
            raise ValueError("background or supplemental sampling differs between branches")
    if any(result["source_progress"] != results["a"]["source_progress"] for result in results.values()):
        raise ValueError("the branches consumed different background populations")
    return {"schema_version": "intrep.rule_transfer_training_audit.v1", "verified": True,
            "updates_per_condition": count, "background_sources_per_update": 12,
            "tuition_examples_per_condition": tuition_count, "tuition_repetitions_per_pair": tuition_count // 90,
            "tuition_yes_no_counts_per_condition": {"yes": tuition_count // 2, "no": tuition_count // 2},
            "initial_checkpoint_sha256": common["checkpoint_sha256"],
            "matched_initial_parameters": True, "matched_background_and_lesson_inputs": True,
            "all_parameters_trainable": True, "new_rule_image_training_examples": 0,
            "prerequisites_passed": all(result["prerequisites_passed"] for result in results.values()),
            "training_seconds": {"calibration": common["training_seconds"],
                                  **{name: result["training_seconds"] for name, result in results.items()}},
            "scope": "Recorded configurations, reader states, input traces and training-code modality contract; not a proof of transfer."}


def read_trial(directory):
    result = json.loads((directory / "result.json").read_text())
    path = directory / "steps.jsonl"
    trace = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    return result, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True)
    for name in ("a", "b", "control"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common, common_trace = read_trial(args.common)
    records = {name: read_trial(getattr(args, name)) for name in ("a", "b", "control")}
    result = audit_conditions(common, {name: value[0] for name, value in records.items()},
                              {name: value[1] for name, value in records.items()}, common_trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
