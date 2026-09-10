"""Experimental one-step control with a frozen learned cellular predictor.

Only the executed intervention becomes experience. Counterfactual ground truth
is used after action selection to measure regret, never to choose learned actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from intrep.problems.cellular_rule_inference.episodes import (
    deserialize_rules,
    rule_id,
    sample_rules,
    serialize_rules,
)
from intrep.problems.cellular_rule_inference.stress import baseline_probabilities
from intrep.problems.cellular_rule_inference.training import load_checkpoint
from intrep.representation.assemblies.cellular_rule_inference import (
    CellularRuleInferenceModel,
)
from intrep.worlds.cellular.arrays import step_grids
from intrep.worlds.cellular.world import CellularRule

ROUNDS = 9
CONTEXT_COUNTS = (0, 1, 4, 8)
METHODS = ("experience", "forgetful", "wrong_context", "family_bayes", "no_op", "random", "oracle")


@dataclass(frozen=True)
class ControlTasks:
    boards: np.ndarray  # [round, trial, height, width]
    goals: np.ndarray
    tie_orders: np.ndarray  # [round, trial, action]; fixed independently of outcomes


def intervention_boards(boards: np.ndarray) -> np.ndarray:
    """Action zero waits; action 1+i flips row-major cell i before evolution."""
    if boards.ndim != 3 or not np.isin(boards, (0, 1)).all():
        raise ValueError("boards must be binary [batch, height, width] arrays")
    batch, height, width = boards.shape
    cells = height * width
    result = np.repeat(boards[:, None], cells + 1, axis=1)
    flat = result.reshape(batch, cells + 1, cells)
    flat[:, np.arange(1, cells + 1), np.arange(cells)] ^= 1
    return result


def sample_tasks(rng: np.random.Generator, *, trials: int, height: int, width: int) -> ControlTasks:
    if trials < 1 or height * width < 9:
        raise ValueError("need positive trial count and at least nine cells")
    boards = np.empty((ROUNDS, trials, height, width), dtype=np.int64)
    for trial in range(trials):
        for turn in range(ROUNDS):
            # Distance >=3 makes the entire action-candidate pools disjoint.
            # This rejection depends only on inputs, never rules, actions or scores.
            for _ in range(10000):
                board = (rng.random((height, width)) < rng.choice((0.2, 0.5, 0.8))).astype(np.int64)
                distances = (boards[:turn, trial] != board).sum(axis=(-2, -1))
                if (distances >= 3).all():
                    boards[turn, trial] = board
                    break
            else:
                raise ValueError("could not sample disjoint intervention pools")
    # Goals are independent of the private rule and its possible outcomes.
    goals = rng.integers(0, 2, size=boards.shape)
    actions = height * width + 1
    orders = np.stack([rng.permutation(actions) for _ in range(ROUNDS * trials)])
    return ControlTasks(boards, goals, orders.reshape(ROUNDS, trials, actions))


def expected_matches(probability: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """Expected matching cells; linear utility needs only cell marginals."""
    return np.where(goals[:, None] == 1, probability, 1 - probability).sum(axis=(-2, -1))


def choose_actions(scores: np.ndarray, tie_orders: np.ndarray) -> np.ndarray:
    ordered = np.take_along_axis(scores, tie_orders, axis=1)
    return np.take_along_axis(tie_orders, ordered.argmax(axis=1)[:, None], axis=1)[:, 0]


@torch.inference_mode()
def predict_candidates(model: CellularRuleInferenceModel, support: np.ndarray,
                       candidates: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size < 1 or len(support) != len(candidates):
        raise ValueError("need positive batch size and matching context/candidate batches")
    count = candidates.shape[1]
    queries = candidates.reshape(-1, *candidates.shape[-2:])
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        context_indices = np.arange(start, stop) // count
        logits = model(torch.as_tensor(support[context_indices], dtype=torch.float32, device=device),
                       torch.as_tensor(queries[start:stop], dtype=torch.float32, device=device))
        chunks.append(logits.softmax(-1)[..., 1].cpu().numpy())
    return np.concatenate(chunks).reshape(candidates.shape)


def decision_metrics(actual_scores: np.ndarray, choices: dict[str, np.ndarray]) -> dict:
    best = actual_scores.max(axis=1)
    informative = best > actual_scores.min(axis=1)
    methods = {}
    for name in METHODS:
        if name == "random":
            optimal = (actual_scores == best[:, None]).mean(axis=1)
            regret = (best[:, None] - actual_scores).mean(axis=1)
            gain = (actual_scores - actual_scores[:, :1]).mean(axis=1)
        elif name == "oracle":
            optimal = np.ones(len(best))
            regret = np.zeros(len(best))
            gain = best - actual_scores[:, 0]
        else:
            obtained = actual_scores[np.arange(len(best)), choices[name]]
            optimal = obtained == best
            regret = best - obtained
            gain = obtained - actual_scores[:, 0]
        methods[name] = {
            "optimal_count": float(optimal[informative].sum()),
            "regret_sum": float(regret[informative].sum()),
            "gain_over_noop_sum": float(gain[informative].sum()),
        }
    return {"tasks": len(best), "decision_tasks": int(informative.sum()), "methods": methods}


def run_tasks(model: CellularRuleInferenceModel, rule: CellularRule, donor: CellularRule,
              tasks: ControlTasks, *, batch_size: int = 64, retain_trace: bool = False) -> list[dict]:
    """The experience policy controls the rollout; controls share its evidence."""
    model.eval()
    trials, height, width = tasks.boards.shape[1:]
    history = np.empty((trials, 0, 2, height, width), dtype=np.int64)
    rows = []
    for turn, (boards, goals, orders) in enumerate(zip(tasks.boards, tasks.goals, tasks.tie_orders)):
        count = max(k for k in CONTEXT_COUNTS if k <= turn)
        support = history[:, -count:] if count else history[:, :0]
        candidates = intervention_boards(boards)
        probability = predict_candidates(model, support, candidates, batch_size)
        empty = predict_candidates(model, history[:, :0], candidates, batch_size) if count else probability
        if count:
            wrong = support.copy()
            wrong[:, :, 1] = step_grids(wrong[:, :, 0], [donor] * trials)
            wrong_probability = predict_candidates(model, wrong, candidates, batch_size)
        else:
            wrong_probability = probability
        repeated_support = np.repeat(support, candidates.shape[1], axis=0)
        flattened_candidates = candidates.reshape(-1, height, width)
        family_probability = baseline_probabilities(repeated_support, flattened_candidates, 0)["family_bayes"]
        family_probability = family_probability.reshape(candidates.shape)
        choices = {name: choose_actions(expected_matches(p, goals), orders) for name, p in (
            ("experience", probability), ("forgetful", empty),
            ("wrong_context", wrong_probability), ("family_bayes", family_probability),
        )}
        choices["no_op"] = np.zeros(trials, dtype=np.int64)

        # The learner receives only its own executed transition. The evaluator
        # additionally enumerates unexecuted outcomes after decisions are fixed.
        selected = candidates[np.arange(trials), choices["experience"]]
        observed = step_grids(selected, [rule] * trials)
        all_outcomes = step_grids(candidates, [rule] * trials)
        actual_scores = expected_matches(all_outcomes, goals)
        row = {"round": turn, "context_count": count, **decision_metrics(actual_scores, choices)}
        if retain_trace:
            trial = 0
            action = int(choices["experience"][trial])
            forgetful_action = int(choices["forgetful"][trial])
            row["trace"] = {
                "board": boards[trial].tolist(), "goal": goals[trial].tolist(),
                "experience_action": action, "forgetful_action": forgetful_action,
                "intervened_board": selected[trial].tolist(), "observed": observed[trial].tolist(),
                "predicted_probability": probability[trial, action].tolist(),
                "forgetful_observed": all_outcomes[trial, forgetful_action].tolist(),
                "action_scores": actual_scores[trial].tolist(),
                "used_experience_rounds": list(range(turn - count, turn)),
            }
        rows.append(row)
        transition = np.stack((selected, observed), axis=1)[:, None]
        history = np.concatenate((history, transition), axis=1)
    return rows


def summarize_round(rows: list[dict]) -> dict:
    counts = np.array([row["decision_tasks"] for row in rows])
    total = int(counts.sum())
    draws = np.random.default_rng(813).integers(0, len(rows), size=(2000, len(rows)))
    denominators = counts[draws].sum(axis=1)

    def estimate(sums: np.ndarray) -> dict:
        if total == 0:
            return {"mean": None, "rule_pair_bootstrap_ci95": None}
        valid = denominators > 0
        bootstrap = sums[draws].sum(axis=1)[valid] / denominators[valid]
        return {"mean": float(sums.sum() / total),
                "rule_pair_bootstrap_ci95": np.quantile(bootstrap, (0.025, 0.975)).tolist()}

    summary = {"round": rows[0]["round"], "context_count": rows[0]["context_count"],
               "tasks": sum(row["tasks"] for row in rows), "decision_tasks": total, "methods": {}}
    for method in METHODS:
        metrics = {}
        for output, source in (("optimal_action_rate", "optimal_count"),
                               ("regret_cells", "regret_sum"),
                               ("gain_over_noop_cells", "gain_over_noop_sum")):
            metrics[output] = estimate(np.array([row["methods"][method][source] for row in rows]))
        summary["methods"][method] = metrics
    summary["paired_effects"] = {}
    for control in ("forgetful", "wrong_context"):
        summary["paired_effects"][control] = {
            "regret_reduction_cells": estimate(np.array([
                row["methods"][control]["regret_sum"] - row["methods"]["experience"]["regret_sum"]
                for row in rows])),
            "optimal_action_rate_gain": estimate(np.array([
                row["methods"]["experience"]["optimal_count"] - row["methods"][control]["optimal_count"]
                for row in rows])),
        }
    return summary


def evaluate_control(checkpoint: Path, *, rule_count: int = 64, trials_per_rule: int = 4,
                     rule_seed: int = 59101, data_seed: int = 63001, device: str = "cpu",
                     batch_size: int = 64, exclude_rules_from: tuple[Path, ...] = ()) -> dict:
    if rule_count < 2 or trials_per_rule < 1 or batch_size < 1:
        raise ValueError("need at least two rule pairs, positive trials and batch size")
    model, payload = load_checkpoint(checkpoint, device)
    if model.config.max_context != 8:
        raise ValueError("control protocol requires an eight-example checkpoint")
    train_rules = deserialize_rules(payload["train_rules"])
    excluded = train_rules.copy()
    for path in exclude_rules_from:
        excluded.extend(deserialize_rules(json.loads(path.read_text())["eval_rules"]))
    rules = sample_rules(2 * rule_count, rule_seed, excluded=tuple(excluded))
    rng = np.random.default_rng(data_seed)
    task_hash = hashlib.sha256()
    per_rule = []
    for index in range(rule_count):
        rule, donor = rules[2 * index:2 * index + 2]
        tasks = sample_tasks(rng, trials=trials_per_rule, height=model.config.height, width=model.config.width)
        for array in (tasks.boards, tasks.goals, tasks.tie_orders):
            task_hash.update(array.astype("<i8").tobytes())
        rows = run_tasks(model, rule, donor, tasks, batch_size=batch_size, retain_trace=index < 4)
        per_rule.append({"rule_id": rule_id(rule), "donor_rule_id": rule_id(donor), "rounds": rows})
        if (index + 1) % 8 == 0 or index + 1 == rule_count:
            print(json.dumps({"evaluated_control_rule_pairs": index + 1, "total": rule_count}), flush=True)
    return {
        "schema_version": "intrep.cellular_rule_control.v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_step": payload["step"], "training_config": payload["config"],
        "eval_rule_seed": rule_seed, "eval_data_seed": data_seed,
        "trials_per_rule": trials_per_rule, "rounds": ROUNDS,
        "task_sha256": task_hash.hexdigest(),
        "eval_rules": serialize_rules(rules), "train_rule_ids": [rule_id(r) for r in train_rules],
        "excluded_rule_ids": sorted({rule_id(r) for r in excluded}),
        "protocol": {
            "actions": "Wait or flip one cell, then let the unknown rule evolve once.",
            "utility": "Number of cells matching an independently sampled binary goal; exact goals need not be reachable.",
            "memory": "Only the executed before/after transition; no weight updates. New board and goal each round, same private rule.",
            "context": "Latest 0/1/4/8 examples, using the largest trained length not exceeding available experience.",
            "controls": "Decision controls share the experience policy's tasks and evidence; they do not collect their own histories.",
            "headlines": "Only tasks with unequal true action utilities. Random is an exact expectation over all actions; oracle is evaluator-only.",
            "privileged_structure": "Known intervention mechanics, explicit goal utility and exhaustive one-step search. Family Bayes additionally knows the cellular rule family.",
        },
        "summaries": [summarize_round([row["rounds"][turn] for row in per_rule]) for turn in range(ROUNDS)],
        "per_rule": per_rule,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure whether executed experience improves one-step cellular control.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rule-count", type=int, default=64)
    parser.add_argument("--trials-per-rule", type=int, default=4)
    parser.add_argument("--rule-seed", type=int, default=59101)
    parser.add_argument("--data-seed", type=int, default=63001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--exclude-rules-from", type=Path, action="append", default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    result = evaluate_control(args.checkpoint, rule_count=args.rule_count, trials_per_rule=args.trials_per_rule,
                              rule_seed=args.rule_seed, data_seed=args.data_seed, device=args.device,
                              batch_size=args.batch_size, exclude_rules_from=tuple(args.exclude_rules_from))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"final_round": result["summaries"][-1]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
