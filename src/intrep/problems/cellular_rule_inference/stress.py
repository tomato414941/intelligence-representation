"""Bounded noise and chronological rule-change evaluation; no weight updates."""

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
from intrep.problems.cellular_rule_inference.evaluate import bootstrap_interval
from intrep.problems.cellular_rule_inference.training import load_checkpoint
from intrep.worlds.cellular.arrays import neighborhood_keys

NOISE_RATES = (0.0, 0.05, 0.1, 0.2)
CONTEXT_COUNTS = (0, 1, 4, 8)
POST_CHANGE_COUNTS = (0, 1, 2, 4, 6, 8)


def corrupt_outputs(support: np.ndarray, uniforms: np.ndarray, rate: float) -> np.ndarray:
    """Common random numbers keep corruption nested across noise rates/counts."""
    if not 0 <= rate < 0.5 or uniforms.shape != support[:, :, 1].shape:
        raise ValueError("need a noise rate in [0, 0.5) and one draw per output cell")
    result = support.copy()
    result[:, :, 1] ^= (uniforms < rate).astype(result.dtype)
    return result


def splice_context(old: np.ndarray, new: np.ndarray, post_change_count: int) -> np.ndarray:
    """Keep a fixed window: old-rule prefix, then current-rule suffix."""
    if old.shape != new.shape or not 0 <= post_change_count <= old.shape[1]:
        raise ValueError("incompatible contexts or change position")
    if not np.array_equal(old[:, :, 0], new[:, :, 0]):
        raise ValueError("rule changes must keep demonstration inputs fixed")
    result = old.copy()
    if post_change_count:
        result[:, -post_change_count:] = new[:, -post_change_count:]
    return result


def baseline_probabilities(support: np.ndarray, query: np.ndarray, noise_rate: float) -> dict[str, np.ndarray]:
    """Family Bayes knows geometry and sensor noise, but no rule or change point.

    Each unknown rule bit has prior probability 1/2. Its point prediction is
    local majority vote (ties choose zero). B0 is fixed by the world family.
    Frequency uses a Beta(1,1) predictive mean conditional on current cell only.
    """
    if not 0 <= noise_rate < 0.5:
        raise ValueError("noise rate must be in [0, 0.5)")
    query_keys = neighborhood_keys(query)
    context_keys = neighborhood_keys(support[:, :, 0])
    frequency = np.full(query.shape, 0.5)
    family = np.full(query.shape, 0.5)
    for batch in range(len(query)):
        before, after = support[batch, :, 0], support[batch, :, 1]
        for state in (0, 1):
            mask = before == state
            frequency[batch][query[batch] == state] = (after[mask].sum() + 1) / (mask.sum() + 2)
        for key in range(18):
            values = after[context_keys[batch] == key]
            margin = int(2 * values.sum()) - values.size
            if noise_rate == 0:
                # The zero-noise limit also defines a vote for inconsistent histories.
                probability = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            else:
                odds = np.clip(margin * np.log((1 - noise_rate) / noise_rate), -700, 700)
                probability = float(1 / (1 + np.exp(-odds)))
            family[batch][query_keys[batch] == key] = 0.0 if key == 0 else probability
    return {"frequency": frequency, "family_bayes": family}


def probability_metrics(probability: np.ndarray, target: np.ndarray, affected: np.ndarray,
                        identifiable: np.ndarray) -> dict:
    correct = (probability > 0.5) == target
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    out = {"accuracy": float(correct.mean()),
           "nll": float(-(target * np.log(clipped) + (1 - target) * np.log1p(-clipped)).mean()),
           "brier": float(((probability - target) ** 2).mean()), "cells": int(target.size)}
    for name, mask in (("affected", affected), ("identifiable_affected", affected & identifiable)):
        out[name + "_cells"] = int(mask.sum())
        out[name + "_correct"] = int(correct[mask].sum())
    return out


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in rows[0]["methods"]:
        items = [row["methods"][method] for row in rows]
        summary = {metric: float(np.mean([item[metric] for item in items]))
                   for metric in ("accuracy", "nll", "brier")}
        summary["accuracy_ci95"] = bootstrap_interval([item["accuracy"] for item in items])
        summary["cells"] = sum(item["cells"] for item in items)
        for name in ("affected", "identifiable_affected"):
            cells = sum(item[name + "_cells"] for item in items)
            correct = sum(item[name + "_correct"] for item in items)
            summary[name + "_cells"] = cells
            summary[name + "_accuracy"] = correct / cells if cells else None
        result[method] = summary
    return result


def evaluate_stress(checkpoint: Path, *, rule_count: int = 64, queries_per_rule: int = 8,
                    rule_seed: int = 39101, data_seed: int = 43001, device: str = "cpu",
                    exclude_rules_from: tuple[Path, ...] = ()) -> dict:
    if rule_count < 2 or queries_per_rule < 1:
        raise ValueError("need at least two independent rule pairs and one query each")
    model, payload = load_checkpoint(checkpoint, device)
    if model.config.max_context != 8:
        raise ValueError("stress protocol requires eight-example contexts")
    training_rules = deserialize_rules(payload["train_rules"])
    excluded = training_rules.copy()
    for source in exclude_rules_from:
        excluded.extend(deserialize_rules(json.loads(source.read_text())["eval_rules"]))
    # Disjoint pairs make rule-pair bootstrap samples independent.
    rules = sample_rules(2 * rule_count, rule_seed, excluded=tuple(excluded))
    resolved = next(model.parameters()).device
    model.eval()
    rng = np.random.default_rng(data_seed)
    noise_rows = {(rate, count): [] for rate in NOISE_RATES for count in CONTEXT_COUNTS}
    change_rows = {(rate, count): [] for rate in (0.0, 0.1, 0.2) for count in POST_CHANGE_COUNTS}

    def predict(support, query):
        logits = model(torch.as_tensor(support, dtype=torch.float32, device=resolved),
                       torch.as_tensor(query, dtype=torch.float32, device=resolved))
        return logits.softmax(-1)[..., 1].cpu().numpy().reshape(query.shape)

    with torch.inference_mode():
        for index in range(rule_count):
            old_rule, new_rule = rules[2 * index:2 * index + 2]
            old = sample_episodes([old_rule] * queries_per_rule, rng, height=model.config.height,
                                  width=model.config.width, context_count=8)
            new = replace_context_rules(old, [new_rule] * queries_per_rule)
            uniforms = rng.random(new.support[:, :, 1].shape)
            affected = old.targets != new.targets
            identity = {"old_rule_id": rule_id(old_rule), "new_rule_id": rule_id(new_rule)}
            stable_predictions = {}
            for (rate, count), rows in noise_rows.items():
                support = corrupt_outputs(new.support, uniforms, rate)[:, :count]
                probabilities = baseline_probabilities(support, new.query, rate)
                probabilities["model"] = predict(support, new.query)
                if count == 8:
                    stable_predictions[rate] = probabilities["model"]
                covered = evidence_coverage(support, new.query)
                rows.append(identity | {"methods": {
                    name: probability_metrics(p, new.targets, affected, covered)
                    for name, p in probabilities.items()}})
            for (rate, count), rows in change_rows.items():
                support = corrupt_outputs(splice_context(old.support, new.support, count), uniforms, rate)
                current = support[:, 8 - count:]
                probabilities = baseline_probabilities(support, new.query, rate)
                for recent in (2, 4):
                    probabilities[f"family_recent{recent}"] = baseline_probabilities(
                        support[:, -recent:], new.query, rate)["family_bayes"]
                probabilities["model"] = stable_predictions[rate] if count == 8 else predict(support, new.query)
                probabilities["model_recent2"] = predict(support[:, -2:], new.query)
                # Privileged boundary control; the main model never receives the boundary.
                probabilities["model_current_only"] = predict(current, new.query)
                probabilities["model_stable"] = stable_predictions[rate]
                probabilities["model_reversed"] = predict(support[:, ::-1].copy(), new.query)
                covered = evidence_coverage(current, new.query)
                rows.append(identity | {"methods": {
                    name: probability_metrics(p, new.targets, affected, covered)
                    for name, p in probabilities.items()}})
            if (index + 1) % 16 == 0:
                print(json.dumps({"evaluated_rule_pairs": index + 1, "total": rule_count}), flush=True)

    def summarize(groups, count_name):
        return [{"noise_rate": rate, count_name: count, "methods": aggregate(rows), "per_pair": rows}
                for (rate, count), rows in groups.items()]

    return {"schema_version": "intrep.cellular_rule_stress.v1",
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_step": payload["step"], "training_config": payload["config"],
            "eval_rules": serialize_rules(rules), "excluded_rule_ids": sorted({rule_id(r) for r in excluded}),
            "rule_seed": rule_seed, "data_seed": data_seed, "rule_pairs": rule_count,
            "queries_per_rule": queries_per_rule,
            "protocol": "Independent before states, chronological outputs, fixed eight-example change windows. "
                        "Only observed outputs are corrupted; queries/targets are clean. Weights frozen. "
                        "Zero post-change observations is unobservable and is not an adaptation failure. "
                        "Bootstrap resamples disjoint rule pairs. NLL probabilities clipped to [1e-7,1-1e-7].",
            "noise": summarize(noise_rows, "context_count"),
            "change": summarize(change_rows, "post_change_count")}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-count", type=int, default=64)
    parser.add_argument("--queries-per-rule", type=int, default=8)
    parser.add_argument("--rule-seed", type=int, default=39101)
    parser.add_argument("--data-seed", type=int, default=43001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--exclude-rules-from", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    result = evaluate_stress(args.checkpoint, rule_count=args.rule_count, queries_per_rule=args.queries_per_rule,
                             rule_seed=args.rule_seed, data_seed=args.data_seed, device=args.device,
                             exclude_rules_from=tuple(args.exclude_rules_from))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
