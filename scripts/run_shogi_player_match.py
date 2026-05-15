from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from intrep.problems.shogi_policy_value.generated_data_cycle import DEFAULT_SHOGI_MAX_PLIES


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a shogi player-vs-player match.")
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    _add_player_arguments(parser, "player-a")
    _add_player_arguments(parser, "player-b")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=64)
    parser.add_argument("--move-time-limit-sec", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    args = parser.parse_args(argv)

    _validate_player_arguments(parser, args, "player_a")
    _validate_player_arguments(parser, args, "player_b")

    subprocess.run(build_shogi_player_match_command(args), cwd=args.arena_repo.resolve(), check=True)


def _add_player_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    parser.add_argument(f"--{prefix}-kind", choices=("checkpoint", "usi", "deterministic_legal"), default="checkpoint")
    parser.add_argument(f"--{prefix}-checkpoint", type=Path)
    parser.add_argument(f"--{prefix}-usi-command")
    parser.add_argument(f"--{prefix}-usi-option", action="append", default=[])
    parser.add_argument(f"--{prefix}-usi-go-command", default="go nodes 1")


def _validate_player_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, name: str) -> None:
    kind = getattr(args, f"{name}_kind")
    flag_prefix = name.replace("_", "-")
    if kind == "checkpoint" and getattr(args, f"{name}_checkpoint") is None:
        parser.error(f"--{flag_prefix}-checkpoint is required when --{flag_prefix}-kind checkpoint")
    if kind == "usi" and not getattr(args, f"{name}_usi_command"):
        parser.error(f"--{flag_prefix}-usi-command is required when --{flag_prefix}-kind usi")


def build_shogi_player_match_command(args: argparse.Namespace) -> list[str]:
    return [
        *_shogi_arena_python_command(),
        "scripts/evaluate_shogi_players.py",
        *_arena_player_args(args, source_prefix="player_a", arena_prefix="player-a"),
        *_arena_player_args(args, source_prefix="player_b", arena_prefix="player-b"),
        "--games",
        str(args.games),
        "--max-plies",
        str(args.max_plies),
        "--out",
        str(args.out.resolve()),
    ]


def _arena_player_args(args: argparse.Namespace, *, source_prefix: str, arena_prefix: str) -> list[str]:
    kind = getattr(args, f"{source_prefix}_kind")
    command = [f"--{arena_prefix}-kind", _arena_kind(kind)]
    if kind == "checkpoint":
        command.extend(
            [
                f"--{arena_prefix}-checkpoint",
                str(getattr(args, f"{source_prefix}_checkpoint").resolve()),
                f"--{arena_prefix}-move-selection-profile",
                "evaluation",
                f"--{arena_prefix}-move-selector",
                "mcts",
                f"--{arena_prefix}-mcts-simulations",
                str(args.simulations),
                f"--{arena_prefix}-mcts-evaluation-batch-size",
                str(args.evaluation_batch_size),
                f"--{arena_prefix}-device",
                args.device,
                f"--{arena_prefix}-board-backend",
                args.board_backend,
            ]
        )
        if args.move_time_limit_sec is not None:
            command.extend([f"--{arena_prefix}-mcts-move-time-limit-sec", str(args.move_time_limit_sec)])
    elif kind == "usi":
        command.extend(
            [
                f"--{arena_prefix}-usi-command",
                getattr(args, f"{source_prefix}_usi_command"),
                f"--{arena_prefix}-usi-go-command",
                getattr(args, f"{source_prefix}_usi_go_command"),
            ]
        )
        for option in getattr(args, f"{source_prefix}_usi_option"):
            command.extend([f"--{arena_prefix}-usi-option", option])
    return command


def _arena_kind(kind: str) -> str:
    if kind == "deterministic_legal":
        return "deterministic_legal"
    if kind in {"checkpoint", "usi"}:
        return kind
    raise ValueError(f"unsupported player kind: {kind}")


def _shogi_arena_python_command() -> list[str]:
    python = os.environ.get("SHOGI_ARENA_PYTHON")
    if python:
        return [python]
    return ["uv", "run", "python"]


if __name__ == "__main__":
    main()
