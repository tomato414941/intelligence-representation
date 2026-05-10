from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one manual shogi RL data-generation and training cycle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument("--opponent", choices=("self", "yaneuraou"), default="self")
    parser.add_argument("--yaneuraou", help="USI engine command used when --opponent yaneuraou.")
    parser.add_argument("--engine-go-command", default="go nodes 1")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    games_jsonl = run_dir / "generated-games.jsonl"
    train_jsonl = run_dir / "train-games.jsonl"
    eval_jsonl = run_dir / "eval-games.jsonl"
    data_selection_json = run_dir / "data-selection.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best-checkpoint.pt"
    metrics_path = run_dir / "metrics.json"

    _run_generate_games(
        arena_repo=args.arena_repo,
        checkpoint=args.checkpoint,
        opponent=args.opponent,
        yaneuraou=args.yaneuraou,
        engine_go_command=args.engine_go_command,
        out=games_jsonl,
        games=args.games,
        max_plies=args.max_plies,
        simulations=args.simulations,
        evaluation_batch_size=args.evaluation_batch_size,
    )
    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=games_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        eval_ratio=args.eval_ratio,
    )
    _write_data_selection(data_selection_json, train_jsonl=train_jsonl, eval_jsonl=eval_jsonl)
    _run_training(
        data_selection_json=data_selection_json,
        init_checkpoint_path=args.checkpoint,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "generated_games_jsonl": str(games_jsonl),
                "train_games": train_count,
                "eval_games": eval_count,
                "data_selection": str(data_selection_json),
                "checkpoint": str(checkpoint_path),
                "best_checkpoint": str(best_checkpoint_path),
                "metrics": str(metrics_path),
            },
            indent=2,
        )
    )


def _run_generate_games(
    *,
    arena_repo: Path,
    checkpoint: Path,
    opponent: str,
    yaneuraou: str | None,
    engine_go_command: str,
    out: Path,
    games: int,
    max_plies: int,
    simulations: int,
    evaluation_batch_size: int,
) -> None:
    if opponent == "yaneuraou" and not yaneuraou:
        raise SystemExit("--yaneuraou is required when --opponent yaneuraou")

    command = [
        "uv",
        "run",
        "python",
        "scripts/generate_shogi_games.py",
        "--black-kind",
        "checkpoint",
        "--black-checkpoint",
        str(checkpoint.resolve()),
        "--black-checkpoint-policy",
        "mcts",
        "--black-checkpoint-simulations",
        str(simulations),
        "--black-checkpoint-evaluation-batch-size",
        str(evaluation_batch_size),
        "--games",
        str(games),
        "--max-plies",
        str(max_plies),
        "--out",
        str(out),
    ]
    if opponent == "yaneuraou":
        command.extend(
            [
                "--white-kind",
                "yaneuraou",
                "--white-yaneuraou-command",
                yaneuraou or "",
                "--white-yaneuraou-go-command",
                engine_go_command,
            ]
        )
    else:
        command.extend(
            [
                "--white-kind",
                "checkpoint",
                "--white-checkpoint",
                str(checkpoint.resolve()),
                "--white-checkpoint-policy",
                "mcts",
                "--white-checkpoint-simulations",
                str(simulations),
                "--white-checkpoint-evaluation-batch-size",
                str(evaluation_batch_size),
            ]
        )
    subprocess.run(command, cwd=arena_repo.resolve(), check=True)


def _write_data_selection(path: Path, *, train_jsonl: Path, eval_jsonl: Path) -> None:
    payload = {
        "name": path.parent.name,
        "objective": "shogi policy/value from self-play records",
        "target_construction": {
            "policy": "chosen_move",
            "policy_temperature_cp": 100.0,
            "policy_mate_cp": 100000.0,
            "value": "winner",
            "score_cp_scale": 600.0,
        },
        "train_sources": [{"kind": "game_records_jsonl", "path": str(train_jsonl)}],
        "eval_sources": [{"kind": "game_records_jsonl", "path": str(eval_jsonl)}],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_training(
    *,
    data_selection_json: Path,
    init_checkpoint_path: Path,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    metrics_path: Path,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    policy_loss_weight: float,
    value_loss_weight: float,
    device: str,
    num_workers: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "intrep.train_shogi_policy_value",
        "--data-selection",
        str(data_selection_json),
        "--init-checkpoint-path",
        str(init_checkpoint_path),
        "--checkpoint-path",
        str(checkpoint_path),
        "--best-checkpoint-path",
        str(best_checkpoint_path),
        "--metrics-path",
        str(metrics_path),
        "--max-steps",
        str(max_steps),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--policy-loss-weight",
        str(policy_loss_weight),
        "--value-loss-weight",
        str(value_loss_weight),
        "--device",
        device,
        "--num-workers",
        str(num_workers),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
