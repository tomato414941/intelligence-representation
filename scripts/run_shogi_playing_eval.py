from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from intrep.problems.shogi_policy_value.generated_data_cycle import DEFAULT_SHOGI_MAX_PLIES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run shogi playing evaluation for a checkpoint.")
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--opponent-kind", choices=("checkpoint", "yaneuraou", "deterministic_legal"), default="yaneuraou")
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--yaneuraou", help="USI engine command used when --opponent-kind yaneuraou.")
    parser.add_argument("--engine-go-command", default="go nodes 1")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--move-time-limit-sec", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    args = parser.parse_args(argv)

    if args.opponent_kind == "checkpoint" and args.opponent_checkpoint is None:
        parser.error("--opponent-checkpoint is required when --opponent-kind checkpoint")
    if args.opponent_kind == "yaneuraou" and not args.yaneuraou:
        parser.error("--yaneuraou is required when --opponent-kind yaneuraou")

    subprocess.run(build_shogi_playing_eval_command(args), cwd=args.arena_repo.resolve(), check=True)


def build_shogi_playing_eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        *_shogi_arena_python_command(),
        "scripts/evaluate_shogi_players.py",
        "--player-kind",
        "checkpoint",
        "--player-checkpoint",
        str(args.checkpoint.resolve()),
        "--player-checkpoint-profile",
        "evaluation",
        "--player-checkpoint-policy",
        "mcts",
        "--player-checkpoint-simulations",
        str(args.simulations),
        "--player-checkpoint-evaluation-batch-size",
        str(args.evaluation_batch_size),
        "--player-checkpoint-device",
        args.device,
        "--player-checkpoint-board-backend",
        args.board_backend,
        "--games",
        str(args.games),
        "--max-plies",
        str(args.max_plies),
        "--out",
        str(args.out.resolve()),
    ]
    if args.move_time_limit_sec is not None:
        command.extend(["--player-checkpoint-move-time-limit-sec", str(args.move_time_limit_sec)])
    if args.opponent_kind == "checkpoint":
        command.extend(
            [
                "--opponent-kind",
                "checkpoint",
                "--opponent-checkpoint",
                str(args.opponent_checkpoint.resolve()),
                "--opponent-checkpoint-profile",
                "evaluation",
                "--opponent-checkpoint-policy",
                "mcts",
                "--opponent-checkpoint-simulations",
                str(args.simulations),
                "--opponent-checkpoint-evaluation-batch-size",
                str(args.evaluation_batch_size),
                "--opponent-checkpoint-device",
                args.device,
                "--opponent-checkpoint-board-backend",
                args.board_backend,
            ]
        )
        if args.move_time_limit_sec is not None:
            command.extend(["--opponent-checkpoint-move-time-limit-sec", str(args.move_time_limit_sec)])
    elif args.opponent_kind == "yaneuraou":
        command.extend(
            [
                "--opponent-kind",
                "yaneuraou",
                "--opponent-yaneuraou-command",
                args.yaneuraou,
                "--opponent-yaneuraou-go-command",
                args.engine_go_command,
            ]
        )
    else:
        command.extend(["--opponent-kind", "deterministic_legal"])
    return command


def _shogi_arena_python_command() -> list[str]:
    python = os.environ.get("SHOGI_ARENA_PYTHON")
    if python:
        return [python]
    return ["uv", "run", "python"]


if __name__ == "__main__":
    main()
