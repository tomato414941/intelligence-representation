from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.problems.cellular_step_prediction.checkpoint import save_cellular_step_checkpoint
from intrep.problems.cellular_step_prediction.training import (
    CellularStepPredictionConfig,
    train_cellular_step_predictor,
)
from intrep.worlds.cellular.world import LIFE_RULE, generate_cellular_transitions, generate_random_cellular_rule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a cellular next-observation predictor. Training only: data is "
            "declared by generation parameters, and evaluation is a separate command."
        ),
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--rule-seed", type=int, help="Sample a random rule; default is Life (B3/S23).")
    parser.add_argument("--grid-width", type=int, default=6)
    parser.add_argument("--grid-height", type=int, default=6)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--train-seed", type=int, default=1)
    parser.add_argument("--alive-probability", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rule = generate_random_cellular_rule(args.rule_seed) if args.rule_seed is not None else LIFE_RULE
    transitions = generate_cellular_transitions(
        rule,
        width=args.grid_width,
        height=args.grid_height,
        count=args.train_count,
        alive_probability=args.alive_probability,
        seed=args.train_seed,
    )
    config = CellularStepPredictionConfig(
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
    artifacts = train_cellular_step_predictor(transitions, config=config)
    save_cellular_step_checkpoint(args.checkpoint_path, artifacts, rule=rule)
    result = artifacts.result
    payload = {
        "schema_version": "intrep.cellular_step_prediction_run.v1",
        "world": {
            "kind": "cellular",
            "width": args.grid_width,
            "height": args.grid_height,
            "rule": {"birth": sorted(rule.birth), "survival": sorted(rule.survival)},
            "rule_seed": args.rule_seed,
        },
        "train_data": {
            "count": args.train_count,
            "seed": args.train_seed,
            "alive_probability": args.alive_probability,
        },
        "objective": "predict the next cellular observation per cell from the current observation",
        "training_config": asdict(config),
        "checkpoint_path": str(args.checkpoint_path),
        "result": {
            "train_case_count": result.train_case_count,
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
            "train_scores": asdict(result.train_scores),
            "max_steps": result.max_steps,
        },
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    scores = result.train_scores
    print("intrep train cellular step prediction")
    print(
        f"train_cases={result.train_case_count}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" train_changed_cell_accuracy={scores.changed_cell_accuracy if scores.changed_cell_accuracy is not None else 'none'}"
        f" train_unchanged_cell_accuracy={scores.unchanged_cell_accuracy if scores.unchanged_cell_accuracy is not None else 'none'}"
    )


if __name__ == "__main__":
    main()
