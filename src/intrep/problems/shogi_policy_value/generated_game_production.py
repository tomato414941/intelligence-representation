from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STANDARD_SHOGI_MAX_PLIES = 320
DEFAULT_SHOGI_MAX_PLIES = 320
DEFAULT_USI_READ_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class ShogiGeneratedPlayerSpec:
    kind: str
    name: str
    usi_command: str | None = None
    usi_options: tuple[str, ...] = ()
    usi_go_command: str = "go nodes 1"
    usi_read_timeout_seconds: float = DEFAULT_USI_READ_TIMEOUT_SECONDS


def checkpoint_generated_player(name: str = "checkpoint") -> ShogiGeneratedPlayerSpec:
    return ShogiGeneratedPlayerSpec(kind="checkpoint", name=name)


def usi_engine_generated_player(
    *,
    name: str,
    command: str,
    options: tuple[str, ...] = (),
    go_command: str = "go nodes 1",
    read_timeout_seconds: float = DEFAULT_USI_READ_TIMEOUT_SECONDS,
) -> ShogiGeneratedPlayerSpec:
    return ShogiGeneratedPlayerSpec(
        kind="usi_engine",
        name=name,
        usi_command=command,
        usi_options=options,
        usi_go_command=go_command,
        usi_read_timeout_seconds=read_timeout_seconds,
    )


def run_shogi_generated_games(
    *,
    arena_repo: Path,
    checkpoint: Path,
    black_player: ShogiGeneratedPlayerSpec,
    white_player: ShogiGeneratedPlayerSpec,
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
    _validate_player(black_player, side="black")
    _validate_player(white_player, side="white")

    arena_repo = arena_repo.resolve()
    effective_concurrent_games_per_process = _effective_concurrent_games_per_process(
        black_player=black_player,
        white_player=white_player,
        concurrent_games_per_process=concurrent_games_per_process,
    )
    command = [
        sys.executable,
        str(arena_repo / "scripts/generate_shogi_games.py"),
        *_player_command_args(
            "black",
            black_player,
            checkpoint=checkpoint,
            simulations=simulations,
            evaluation_batch_size=evaluation_batch_size,
            checkpoint_device=checkpoint_device,
            board_backend=board_backend,
            mcts_move_time_limit_sec=mcts_move_time_limit_sec,
        ),
        *_player_command_args(
            "white",
            white_player,
            checkpoint=checkpoint,
            simulations=simulations,
            evaluation_batch_size=evaluation_batch_size,
            checkpoint_device=checkpoint_device,
            board_backend=board_backend,
            mcts_move_time_limit_sec=mcts_move_time_limit_sec,
        ),
        "--games",
        str(games),
        "--concurrent-games-per-process",
        str(effective_concurrent_games_per_process),
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


def _validate_player(player: ShogiGeneratedPlayerSpec, *, side: str) -> None:
    if player.kind == "checkpoint":
        return
    if player.kind == "usi_engine":
        if not player.usi_command:
            raise SystemExit(f"{side} usi_engine player requires usi_command")
        return
    raise SystemExit(f"{side} player kind must be checkpoint or usi_engine")


def _player_command_args(
    prefix: str,
    player: ShogiGeneratedPlayerSpec,
    *,
    checkpoint: Path,
    simulations: int,
    evaluation_batch_size: int,
    checkpoint_device: str,
    board_backend: str,
    mcts_move_time_limit_sec: float | None,
) -> list[str]:
    if player.kind == "checkpoint":
        command = [
            f"--{prefix}-kind",
            "checkpoint",
            f"--{prefix}-checkpoint",
            str(checkpoint.resolve()),
            f"--{prefix}-checkpoint-id",
            _checkpoint_actor_id(checkpoint),
            f"--{prefix}-move-selection-profile",
            "self-play",
            f"--{prefix}-move-selector",
            "mcts",
            f"--{prefix}-mcts-simulations",
            str(simulations),
            f"--{prefix}-mcts-evaluation-batch-size",
            str(evaluation_batch_size),
            f"--{prefix}-device",
            checkpoint_device,
            f"--{prefix}-board-backend",
            board_backend,
        ]
        if mcts_move_time_limit_sec is not None:
            command.extend([f"--{prefix}-mcts-move-time-limit-sec", str(mcts_move_time_limit_sec)])
        return command
    command = [
        f"--{prefix}-kind",
        "usi_engine",
        f"--{prefix}-usi-command",
        player.usi_command or "",
        f"--{prefix}-usi-go-command",
        player.usi_go_command,
        f"--{prefix}-usi-read-timeout-seconds",
        str(player.usi_read_timeout_seconds),
    ]
    for option in player.usi_options:
        command.extend([f"--{prefix}-usi-option", option])
    return command


def _effective_concurrent_games_per_process(
    *,
    black_player: ShogiGeneratedPlayerSpec,
    white_player: ShogiGeneratedPlayerSpec,
    concurrent_games_per_process: int,
) -> int:
    if black_player.kind != "checkpoint" or white_player.kind != "checkpoint":
        return 1
    return concurrent_games_per_process


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
