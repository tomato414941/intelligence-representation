from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

STANDARD_SHOGI_MAX_PLIES = 320
DEFAULT_SHOGI_MAX_PLIES = 320


def run_shogi_generated_games(
    *,
    arena_repo: Path,
    checkpoint: Path,
    opponent: str,
    yaneuraou: str | None,
    engine_go_command: str,
    out: Path,
    generation_summary_path: Path | None,
    games: int,
    concurrent_games_per_process: int,
    generation_progress_every_plies: int,
    board_backend: str,
    max_plies: int,
    simulations: int,
    evaluation_batch_size: int,
    generation_worker_processes: int,
    seed: int | None,
    checkpoint_device: str,
    mcts_move_time_limit_sec: float | None,
) -> None:
    if opponent == "yaneuraou" and not yaneuraou:
        raise SystemExit("--yaneuraou is required when --opponent yaneuraou")

    arena_repo = arena_repo.resolve()
    command = [
        *_shogi_arena_python_command(),
        str(arena_repo / "scripts/generate_shogi_games.py"),
        "--black-kind",
        "checkpoint",
        "--black-checkpoint",
        str(checkpoint.resolve()),
        "--black-checkpoint-id",
        _checkpoint_actor_id(checkpoint),
        "--black-move-selection-profile",
        "self-play",
        "--black-move-selector",
        "mcts",
        "--black-mcts-simulations",
        str(simulations),
        "--black-mcts-evaluation-batch-size",
        str(evaluation_batch_size),
        "--black-device",
        checkpoint_device,
        "--black-board-backend",
        board_backend,
        "--games",
        str(games),
        "--concurrent-games-per-process",
        str(concurrent_games_per_process),
        "--generation-worker-processes",
        str(generation_worker_processes),
        "--progress-every-plies",
        str(generation_progress_every_plies),
        "--board-backend",
        board_backend,
        "--max-plies",
        str(max_plies),
        "--out",
        str(out),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if mcts_move_time_limit_sec is not None:
        command.extend(["--black-mcts-move-time-limit-sec", str(mcts_move_time_limit_sec)])
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
                "--white-checkpoint-id",
                _checkpoint_actor_id(checkpoint),
                "--white-move-selection-profile",
                "self-play",
                "--white-move-selector",
                "mcts",
                "--white-mcts-simulations",
                str(simulations),
                "--white-mcts-evaluation-batch-size",
                str(evaluation_batch_size),
                "--white-device",
                checkpoint_device,
                "--white-board-backend",
                board_backend,
            ]
        )
        if mcts_move_time_limit_sec is not None:
            command.extend(["--white-mcts-move-time-limit-sec", str(mcts_move_time_limit_sec)])
    completed = subprocess.run(
        command,
        cwd=arena_repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=_shogi_arena_env(arena_repo),
    )
    stdout = completed.stdout if completed is not None else ""
    if stdout:
        print(stdout, end="")
        if generation_summary_path is not None:
            generation_summary_path.write_text(stdout, encoding="utf-8")


def warn_short_max_plies(max_plies: int) -> None:
    if max_plies < STANDARD_SHOGI_MAX_PLIES:
        print(
            f"warning: max_plies {max_plies} is below the computer-shogi standard cap "
            f"of {STANDARD_SHOGI_MAX_PLIES}; this can create artificial max_plies draws.",
            file=sys.stderr,
        )


def _shogi_arena_python_command() -> list[str]:
    python = os.environ.get("SHOGI_ARENA_PYTHON")
    if python:
        return [python]
    return [sys.executable]


def _shogi_arena_env(arena_repo: Path) -> dict[str, str]:
    pythonpath_parts = [str(arena_repo / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return os.environ | {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}


def _checkpoint_actor_id(checkpoint: Path) -> str:
    if checkpoint.name == "checkpoint.pt" or checkpoint.name == "best-checkpoint.pt":
        return checkpoint.parent.name
    return checkpoint.stem
