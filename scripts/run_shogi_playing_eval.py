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
    parser.add_argument("--opponent-kind", choices=("checkpoint", "usi", "deterministic_legal"), default="usi")
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--usi-command", help="USI engine command used when --opponent-kind usi.")
    parser.add_argument("--usi-option", action="append", default=[], help="USI engine option as NAME=VALUE.")
    parser.add_argument("--usi-go-command", default="go nodes 1")
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
    if args.opponent_kind == "usi" and not args.usi_command:
        parser.error("--usi-command is required when --opponent-kind usi")

    subprocess.run(build_shogi_playing_eval_command(args), cwd=args.arena_repo.resolve(), check=True)


def build_shogi_playing_eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        *_shogi_arena_python_command(),
        "scripts/evaluate_shogi_players.py",
        "--player-kind",
        "checkpoint",
        "--player-checkpoint",
        str(args.checkpoint.resolve()),
        "--player-move-selection-profile",
        "evaluation",
        "--player-move-selector",
        "mcts",
        "--player-mcts-simulations",
        str(args.simulations),
        "--player-mcts-evaluation-batch-size",
        str(args.evaluation_batch_size),
        "--player-device",
        args.device,
        "--player-board-backend",
        args.board_backend,
        "--games",
        str(args.games),
        "--max-plies",
        str(args.max_plies),
        "--out",
        str(args.out.resolve()),
    ]
    if args.move_time_limit_sec is not None:
        command.extend(["--player-mcts-move-time-limit-sec", str(args.move_time_limit_sec)])
    if args.opponent_kind == "checkpoint":
        command.extend(
            [
                "--opponent-kind",
                "checkpoint",
                "--opponent-checkpoint",
                str(args.opponent_checkpoint.resolve()),
                "--opponent-move-selection-profile",
                "evaluation",
                "--opponent-move-selector",
                "mcts",
                "--opponent-mcts-simulations",
                str(args.simulations),
                "--opponent-mcts-evaluation-batch-size",
                str(args.evaluation_batch_size),
                "--opponent-device",
                args.device,
                "--opponent-board-backend",
                args.board_backend,
            ]
        )
        if args.move_time_limit_sec is not None:
            command.extend(["--opponent-mcts-move-time-limit-sec", str(args.move_time_limit_sec)])
    elif args.opponent_kind == "usi":
        command.extend(
            [
                "--opponent-kind",
                "usi",
                "--opponent-usi-command",
                args.usi_command,
                "--opponent-usi-go-command",
                args.usi_go_command,
            ]
        )
        for option in args.usi_option:
            command.extend(["--opponent-usi-option", option])
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
