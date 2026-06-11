from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from intrep.core.training_utils import resolve_training_device
from intrep.problems.cellular_step_prediction.checkpoint import load_cellular_step_checkpoint
from intrep.problems.cellular_step_prediction.dataset import (
    CellularStepPredictionDataset,
    cellular_state_to_cell_ids,
)
from intrep.problems.cellular_step_prediction.metrics import cellular_step_prediction_scores
from intrep.problems.cellular_step_prediction.training import predict_cell_ids
from intrep.worlds.cellular.world import generate_cellular_transitions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a cellular step prediction checkpoint on states it never "
            "trained on. Rejects eval seeds that overlap the training data."
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--alive-probability", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    checkpoint = load_cellular_step_checkpoint(args.checkpoint_path, device=args.device)
    reject_train_eval_overlap(
        checkpoint.train_data,
        eval_seed=args.eval_seed,
        eval_count=args.eval_count,
    )
    height, width = checkpoint.grid_size
    transitions = generate_cellular_transitions(
        checkpoint.rule,
        width=width,
        height=height,
        count=args.eval_count,
        alive_probability=args.alive_probability,
        seed=args.eval_seed,
    )
    dataset = CellularStepPredictionDataset(transitions)
    device = resolve_training_device(args.device)
    model_scores = cellular_step_prediction_scores(
        predict_cell_ids(checkpoint.model, dataset, device, batch_size=args.batch_size),
        transitions,
    )
    copy_predictions = torch.stack(
        [cellular_state_to_cell_ids(transition.state) for transition in transitions]
    )
    copy_scores = cellular_step_prediction_scores(copy_predictions, transitions)
    payload = {
        "schema_version": "intrep.cellular_step_prediction_eval.v1",
        "checkpoint_path": str(args.checkpoint_path),
        "world": {
            "kind": "cellular",
            "width": width,
            "height": height,
            "rule": {"birth": sorted(checkpoint.rule.birth), "survival": sorted(checkpoint.rule.survival)},
        },
        "train_data": checkpoint.train_data,
        "eval_data": {
            "count": args.eval_count,
            "seed": args.eval_seed,
            "alive_probability": args.alive_probability,
        },
        "model_scores": asdict(model_scores),
        "copy_baseline_scores": asdict(copy_scores),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("intrep evaluate cellular step prediction")
    print(
        f"eval_cases={args.eval_count}"
        f" changed_cell_accuracy={model_scores.changed_cell_accuracy}"
        f" unchanged_cell_accuracy={model_scores.unchanged_cell_accuracy}"
        f" per_cell_accuracy={model_scores.per_cell_accuracy:.4f}"
    )
    print(
        "copy baseline:"
        f" changed_cell_accuracy={copy_scores.changed_cell_accuracy}"
        f" unchanged_cell_accuracy={copy_scores.unchanged_cell_accuracy}"
    )


def reject_train_eval_overlap(
    train_data: dict[str, object],
    *,
    eval_seed: int,
    eval_count: int,
) -> None:
    """Initial states are identified by their generation seed, so overlapping
    seed ranges would silently evaluate on training states."""
    train_seed = int(train_data["seed"])
    train_count = int(train_data["count"])
    train_end = train_seed + train_count
    eval_end = eval_seed + eval_count
    if eval_seed < train_end and train_seed < eval_end:
        raise ValueError(
            "eval seeds overlap the training data: "
            f"train uses {train_seed}..{train_end - 1}, eval requested {eval_seed}..{eval_end - 1}"
        )


if __name__ == "__main__":
    main()
