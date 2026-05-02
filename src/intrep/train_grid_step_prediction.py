from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.grid_world import GridWorldState, Position, generate_grid_world_transition_table
from intrep.grid_world_prediction import GridStepPredictionConfig, train_grid_step_predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a grid step predictor on a generated transition table.")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    result = train_grid_step_predictor(examples, config=config)
    payload = {
        "schema_version": "intrep.grid_step_prediction_run.v1",
        "world": {
            "kind": "grid_world",
            "width": state.width,
            "height": state.height,
            "goal": asdict(state.goal),
            "walls": [asdict(wall) for wall in sorted(state.walls, key=lambda position: (position.row, position.col))],
        },
        "objective": "predict next agent cell, reward class, and terminated flag from full grid observation and action id",
        "train_case_count": len(examples),
        "training_config": asdict(config),
        "result": asdict(result),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("intrep train grid step prediction")
    print(
        f"train_cases={result.train_case_count}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" next_cell_accuracy={result.next_cell_accuracy:.4f}"
        f" reward_accuracy={result.reward_accuracy:.4f}"
        f" terminated_accuracy={result.terminated_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
