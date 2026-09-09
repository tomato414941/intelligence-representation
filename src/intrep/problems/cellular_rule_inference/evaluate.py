from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from intrep.problems.cellular_rule_inference.episodes import (
    deserialize_rules,
    evidence_coverage,
    replace_context_rules,
    rule_id,
    sample_episodes,
    sample_rules,
    serialize_rules,
)
from intrep.problems.cellular_rule_inference.training import load_checkpoint
from intrep.worlds.cellular.arrays import neighborhood_keys


def context_baselines(support: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frequency ignores geometry; lookup injects the known local rule family.

    Neither baseline sees the query answer. Unseen outcomes default to zero.
    """
    frequency = np.zeros_like(query)
    lookup = np.zeros_like(query)
    if not support.shape[1]:
        return frequency, lookup
    context_keys = neighborhood_keys(support[:, :, 0])
    query_keys = neighborhood_keys(query)
    for batch in range(len(query)):
        before = support[batch, :, 0]
        after = support[batch, :, 1]
        for state in (0, 1):
            mask = before == state
            if mask.any():
                frequency[batch][query[batch] == state] = int(after[mask].mean() > 0.5)
        for key in range(18):
            mask = context_keys[batch] == key
            if mask.any():
                lookup[batch][query_keys[batch] == key] = after[mask][0]
    return frequency, lookup


def bootstrap_interval(values: list[float]) -> list[float]:
    """Resample whole rules, not correlated cells or query boards."""
    values_array = np.asarray(values)
    rng = np.random.default_rng(813)
    draws = rng.choice(values_array, size=(2000, len(values_array)), replace=True).mean(axis=1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def score(predicted: np.ndarray, target: np.ndarray, query: np.ndarray, covered: np.ndarray) -> dict:
    correct = predicted == target
    changed = target != query

    def average(mask: np.ndarray) -> float | None:
        return float(correct[mask].mean()) if mask.any() else None

    return {"accuracy": float(correct.mean()), "changed_accuracy": average(changed),
            "unchanged_accuracy": average(~changed), "covered_accuracy": average(covered),
            "uncovered_accuracy": average(~covered), "coverage": float(covered.mean()),
            "correct_cells": int(correct.sum()), "cells": int(correct.size),
            "covered_correct_cells": int(correct[covered].sum()), "covered_cells": int(covered.sum()),
            "changed_correct_cells": int(correct[changed].sum()), "changed_cells": int(changed.sum())}


def evaluate(checkpoint: Path, *, rule_count: int = 64, queries_per_rule: int = 8,
             rule_seed: int = 9101, data_seed: int = 12001, device: str = "cpu",
             exclude_rules_from: Path | None = None) -> dict:
    if rule_count < 2 or queries_per_rule < 1:
        raise ValueError("need at least two held-out rules and one query each")
    model, payload = load_checkpoint(checkpoint, device)
    training_rules = deserialize_rules(payload["train_rules"])
    excluded = training_rules.copy()
    if exclude_rules_from is not None:
        excluded += deserialize_rules(json.loads(exclude_rules_from.read_text())["eval_rules"])
    rules = sample_rules(rule_count, rule_seed, excluded=tuple(excluded))
    assert not {rule_id(r) for r in rules} & {rule_id(r) for r in training_rules}
    resolved = next(model.parameters()).device
    cfg = model.config
    model.eval()
    results = {count: [] for count in (0, 1, 4, 8)}
    rng = np.random.default_rng(data_seed)
    with torch.inference_mode():
        for index, rule in enumerate(rules):
            episodes = sample_episodes([rule] * queries_per_rule, rng, height=cfg.height, width=cfg.width, context_count=8)
            donor = rules[(index + 1) % len(rules)]
            wrong = replace_context_rules(episodes, [donor] * queries_per_rule)
            query = torch.as_tensor(episodes.query, device=resolved, dtype=torch.float32)
            for count, context_rows in results.items():
                support = episodes.support[:, :count]
                correct_prediction = model(torch.as_tensor(support, device=resolved, dtype=torch.float32), query).argmax(-1).cpu().numpy().reshape(episodes.query.shape)
                wrong_prediction = model(torch.as_tensor(wrong.support[:, :count], device=resolved, dtype=torch.float32), query).argmax(-1).cpu().numpy().reshape(episodes.query.shape)
                covered = evidence_coverage(support, episodes.query)
                frequency, lookup = context_baselines(support, episodes.query)
                row = {"rule_id": rule_id(rule), "donor_rule_id": rule_id(donor),
                       "correct_context": score(correct_prediction, episodes.targets, episodes.query, covered),
                       "wrong_context": score(wrong_prediction, episodes.targets, episodes.query, covered),
                       "donor_target": score(wrong_prediction, wrong.targets, episodes.query, covered),
                       "frequency_baseline": score(frequency, episodes.targets, episodes.query, covered),
                       "rule_family_lookup": score(lookup, episodes.targets, episodes.query, covered),
                       "prediction_change_rate": float((correct_prediction != wrong_prediction).mean())}
                context_rows.append(row)
    summaries = []
    for count, rows in results.items():
        summary = {"context_count": count}
        for condition in ("correct_context", "wrong_context", "donor_target", "frequency_baseline", "rule_family_lookup"):
            values = [row[condition]["accuracy"] for row in rows]
            total = {key: sum(row[condition][key] for row in rows) for key in
                     ("correct_cells", "cells", "covered_correct_cells", "covered_cells", "changed_correct_cells", "changed_cells")}
            uncovered = total["cells"] - total["covered_cells"]
            unchanged = total["cells"] - total["changed_cells"]
            summary[condition] = {
                "accuracy": float(np.mean(values)), "rule_bootstrap_ci95": bootstrap_interval(values),
                "coverage": total["covered_cells"] / total["cells"],
                "covered_accuracy": total["covered_correct_cells"] / total["covered_cells"] if total["covered_cells"] else None,
                "uncovered_accuracy": (total["correct_cells"] - total["covered_correct_cells"]) / uncovered if uncovered else None,
                "changed_accuracy": total["changed_correct_cells"] / total["changed_cells"] if total["changed_cells"] else None,
                "unchanged_accuracy": (total["correct_cells"] - total["changed_correct_cells"]) / unchanged if unchanged else None,
            }
        zero_rows = results[0]
        gains = [row["correct_context"]["accuracy"] - base["correct_context"]["accuracy"] for row, base in zip(rows, zero_rows)]
        context_effect = [row["correct_context"]["accuracy"] - row["wrong_context"]["accuracy"] for row in rows]
        summary["gain_over_zero"] = {"mean": float(np.mean(gains)), "rule_bootstrap_ci95": bootstrap_interval(gains)}
        summary["correct_minus_wrong"] = {"mean": float(np.mean(context_effect)), "rule_bootstrap_ci95": bootstrap_interval(context_effect)}
        summary["prediction_change_rate"] = float(np.mean([row["prediction_change_rate"] for row in rows]))
        summaries.append(summary)
    return {"schema_version": "intrep.cellular_rule_inference_eval.v1", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_step": payload["step"], "training_config": payload["config"],
            "eval_rule_seed": rule_seed, "eval_data_seed": data_seed, "queries_per_rule": queries_per_rule,
            "excluded_rule_ids": sorted({rule_id(r) for r in excluded}),
            "eval_rules": serialize_rules(rules), "train_rule_ids": [rule_id(r) for r in training_rules],
            "conditions": "Nested contexts and fixed queries; donor changes only demonstration outputs. No test-time learning.",
            "coverage_definition": "Query (cell state, neighbor count) occurs in a demonstration; family-aware analysis only.",
            "summaries": summaries, "per_rule": results}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate unseen-rule inference with nested contexts and counterfactual controls.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-count", type=int, default=64)
    parser.add_argument("--queries-per-rule", type=int, default=8)
    parser.add_argument("--rule-seed", type=int, default=9101)
    parser.add_argument("--data-seed", type=int, default=12001)
    parser.add_argument("--exclude-rules-from", type=Path, help="Exclude a previous validation evaluation's rules from the final test.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = evaluate(args.checkpoint, rule_count=args.rule_count, queries_per_rule=args.queries_per_rule,
                      rule_seed=args.rule_seed, data_seed=args.data_seed, device=args.device, exclude_rules_from=args.exclude_rules_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
