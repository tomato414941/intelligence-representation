from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from intrep.domains.grid.world import GridExperienceTransition, GridWorldState, Position, generate_grid_world_transition_table
from intrep.problems.grid_step_prediction.baselines import (
    copy_baseline,
    fit_per_cell_majority,
    naive_action_apply_baseline,
    per_cell_majority_baseline,
)
from intrep.problems.grid_step_prediction.checkpoint import save_grid_core_checkpoint
from intrep.problems.grid_step_prediction.dataset import split_grid_transitions_by_agent_cell
from intrep.problems.grid_step_prediction.metrics import next_observation_metrics
from intrep.problems.grid_step_prediction.training import GridStepPredictionConfig, train_grid_step_predictor_with_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a grid next-observation predictor on a generated transition table.")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--core-checkpoint-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--eval-agent-cell",
        type=int,
        nargs=2,
        action="append",
        metavar=("ROW", "COL"),
        help="Hold out all transitions whose current agent cell matches ROW COL.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    state = GridWorldState(
        width=3,
        height=2,
        agent=Position(row=0, col=0),
        goal=Position(row=1, col=2),
        walls=frozenset({Position(row=1, col=1)}),
    )
    examples = generate_grid_world_transition_table(state)
    train_examples, eval_examples = split_grid_transitions_by_agent_cell(
        examples,
        held_out_cells=[Position(row=row, col=col) for row, col in args.eval_agent_cell or []],
    )
    config = GridStepPredictionConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )
    artifacts = train_grid_step_predictor_with_artifacts(
        train_examples,
        eval_examples=eval_examples or None,
        config=config,
    )
    result = artifacts.result
    baselines = {
        "train": _baseline_metrics(train_examples, train_examples, width=state.width),
        "eval": _baseline_metrics(train_examples, eval_examples, width=state.width) if eval_examples else None,
    }
    payload = {
        "schema_version": "intrep.grid_step_prediction_run.v2",
        "world": {
            "kind": "grid_world",
            "width": state.width,
            "height": state.height,
            "goal": asdict(state.goal),
            "walls": [asdict(wall) for wall in sorted(state.walls, key=lambda position: (position.row, position.col))],
        },
        "objective": "predict the next observation per cell, plus reward class and terminated flag, from the full grid observation and action id",
        "held_out_agent_cells": [asdict(position) for position in sorted(_held_out_cells(args), key=lambda p: (p.row, p.col))],
        "train_case_count": len(train_examples),
        "eval_case_count": len(eval_examples),
        "training_config": asdict(config),
        "core_checkpoint_path": str(args.core_checkpoint_path) if args.core_checkpoint_path is not None else None,
        "result": asdict(result),
        "baselines": baselines,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.core_checkpoint_path is not None:
        save_grid_core_checkpoint(args.core_checkpoint_path, artifacts)
    print("intrep train grid step prediction")
    print(
        f"train_cases={result.train_case_count}"
        f" eval_cases={result.eval_case_count}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" next_agent_cell_accuracy={result.next_agent_cell_accuracy:.4f}"
        f" changed_cell_accuracy={result.changed_cell_accuracy if result.changed_cell_accuracy is not None else 'none'}"
        f" eval_next_agent_cell_accuracy={result.eval_next_agent_cell_accuracy if result.eval_next_agent_cell_accuracy is not None else 'none'}"
    )
    if baselines["eval"] is not None:
        copy_eval = baselines["eval"]["copy"]
        print(
            "baseline eval next_agent_cell_accuracy:"
            f" copy={copy_eval['next_agent_cell_accuracy']:.4f}"
            f" naive_action_apply={baselines['eval']['naive_action_apply']['next_agent_cell_accuracy']:.4f}"
            f" per_cell_majority={baselines['eval']['per_cell_majority']['next_agent_cell_accuracy']:.4f}"
        )


def _baseline_metrics(
    train_examples: Sequence[GridExperienceTransition],
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
) -> dict[str, dict[str, float | None]]:
    majority_table = fit_per_cell_majority(train_examples)
    predictions = {
        "copy": copy_baseline(examples, width=width),
        "naive_action_apply": naive_action_apply_baseline(examples, width=width),
        "per_cell_majority": per_cell_majority_baseline(majority_table, examples),
    }
    return {
        name: asdict(next_observation_metrics(prediction.class_ids, prediction.agent_scores, examples, width=width))
        for name, prediction in predictions.items()
    }


def _held_out_cells(args: argparse.Namespace) -> list[Position]:
    return [Position(row=row, col=col) for row, col in args.eval_agent_cell or []]


if __name__ == "__main__":
    main()
