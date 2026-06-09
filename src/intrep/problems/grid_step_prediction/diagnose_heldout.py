from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.domains.grid.world import GridWorldState, Position
from intrep.problems.grid_step_prediction.heldout_diagnostics import run_held_out_cell_sweep
from intrep.problems.grid_step_prediction.training import GridStepPredictionConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose held-out agent-cell generalization by sweeping every valid "
            "agent cell as the held-out cell across seeds and dumping per-action "
            "next-cell predictions."
        ),
    )
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
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
    config = GridStepPredictionConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device=args.device,
    )
    runs = run_held_out_cell_sweep(state, seeds=args.seeds, config=config)
    payload = {
        "schema_version": "intrep.grid_step_heldout_diagnostic.v1",
        "world": {
            "kind": "grid_world",
            "width": state.width,
            "height": state.height,
            "goal": asdict(state.goal),
            "walls": [asdict(wall) for wall in sorted(state.walls, key=lambda position: (position.row, position.col))],
        },
        "seeds": list(args.seeds),
        "training_config": asdict(config),
        "runs": [
            {
                "held_out_cell": asdict(run.held_out_cell),
                "seed": run.seed,
                "train_case_count": run.train_case_count,
                "eval_case_count": run.eval_case_count,
                "train_next_cell_accuracy": run.train_next_cell_accuracy,
                "eval_next_cell_accuracy": run.eval_next_cell_accuracy,
                "predictions": [
                    {
                        "agent": asdict(prediction.agent),
                        "action": prediction.action,
                        "true_next": asdict(prediction.true_next),
                        "predicted_next": asdict(prediction.predicted_next),
                        "correct": prediction.correct,
                    }
                    for prediction in run.predictions
                ],
            }
            for run in runs
        ],
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("intrep diagnose grid held-out generalization")
    for run in runs:
        print(
            f"held_out=({run.held_out_cell.row}, {run.held_out_cell.col})"
            f" seed={run.seed}"
            f" train_next_cell_accuracy={run.train_next_cell_accuracy:.4f}"
            f" eval_next_cell_accuracy={run.eval_next_cell_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
