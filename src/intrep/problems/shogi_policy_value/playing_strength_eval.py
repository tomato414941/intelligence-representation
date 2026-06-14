from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

STANDARD_SHOGI_MAX_PLIES = 320
DEFAULT_SHOGI_MAX_PLIES = 320
DEFAULT_OPENING_PLIES = 12
DEFAULT_MCTS_SIMULATIONS = 128
DEFAULT_NN_LEAF_EVAL_BATCH_LIMIT = 64
DEFAULT_MCTS_MOVE_TIME_LIMIT_SEC = 9.0
PLAYING_STRENGTH_PROTOCOL = "random_legal_opening_pair_v1"


@dataclass(frozen=True)
class ShogiPlayingStrengthPlayer:
    kind: Literal["checkpoint", "usi_engine"]
    checkpoint: Path | None = None
    checkpoint_id: str | None = None
    usi_command: str | None = None
    usi_options: tuple[str, ...] = ()
    usi_go_command: str = "go nodes 1"
    usi_read_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class ShogiPlayingStrengthEvalConfig:
    arena_repo: Path
    output_dir: Path
    player_a: ShogiPlayingStrengthPlayer
    player_b: ShogiPlayingStrengthPlayer
    games: int
    start_position_seed: int
    opening_plies: int = DEFAULT_OPENING_PLIES
    max_plies: int = DEFAULT_SHOGI_MAX_PLIES
    simulations: int = DEFAULT_MCTS_SIMULATIONS
    nn_leaf_eval_batch_limit: int = DEFAULT_NN_LEAF_EVAL_BATCH_LIMIT
    mcts_move_time_limit_sec: float = DEFAULT_MCTS_MOVE_TIME_LIMIT_SEC
    device: str = "cpu"
    board_backend: str = "cshogi"
    progress_every_games: int = 1

    @property
    def start_position_count(self) -> int:
        return self.games // 2

    @property
    def games_jsonl(self) -> Path:
        return self.output_dir / "games.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "summary.json"

    @property
    def protocol_json(self) -> Path:
        return self.output_dir / "playing_strength_protocol.json"

    def protocol_payload(self) -> dict[str, object]:
        return {
            "schema_version": "intrep.shogi_playing_strength_eval.v1",
            "protocol": PLAYING_STRENGTH_PROTOCOL,
            "games": self.games,
            "start_position_set": "random-legal-opening",
            "start_position_count": self.start_position_count,
            "start_position_seed": self.start_position_seed,
            "opening_plies": self.opening_plies,
            "max_plies": self.max_plies,
            "move_selection_profile": "max-visit",
            "mcts_simulations": self.simulations,
            "nn_leaf_eval_batch_limit": self.nn_leaf_eval_batch_limit,
            "mcts_move_time_limit_sec": self.mcts_move_time_limit_sec,
            "board_backend": self.board_backend,
            "player_a": _player_payload(self.player_a),
            "player_b": _player_payload(self.player_b),
            "files": {
                "games": self.games_jsonl.name,
                "summary": self.summary_json.name,
                "start_positions": "start_positions.jsonl",
            },
        }


def checkpoint_playing_strength_player(
    checkpoint: str | Path,
    *,
    checkpoint_id: str | None = None,
) -> ShogiPlayingStrengthPlayer:
    return ShogiPlayingStrengthPlayer(
        kind="checkpoint",
        checkpoint=Path(checkpoint),
        checkpoint_id=checkpoint_id,
    )


def usi_engine_playing_strength_player(
    *,
    command: str,
    options: tuple[str, ...] = (),
    go_command: str = "go nodes 1",
    read_timeout_seconds: float = 30.0,
) -> ShogiPlayingStrengthPlayer:
    return ShogiPlayingStrengthPlayer(
        kind="usi_engine",
        usi_command=command,
        usi_options=options,
        usi_go_command=go_command,
        usi_read_timeout_seconds=read_timeout_seconds,
    )


def run_shogi_playing_strength_eval(config: ShogiPlayingStrengthEvalConfig) -> dict[str, object]:
    validate_shogi_playing_strength_eval_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.protocol_json.write_text(
        json.dumps(config.protocol_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        build_shogi_playing_strength_eval_command(config),
        cwd=config.arena_repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=_shogi_arena_env(config.arena_repo),
    )
    config.summary_json.write_text(completed.stdout, encoding="utf-8")
    summary = json.loads(completed.stdout)
    return {
        "protocol": config.protocol_payload(),
        "summary": summary,
        "output_dir": str(config.output_dir),
    }


def build_shogi_playing_strength_eval_command(config: ShogiPlayingStrengthEvalConfig) -> list[str]:
    validate_shogi_playing_strength_eval_config(config)
    command = [
        sys.executable,
        str(config.arena_repo / "scripts/evaluate_shogi_players.py"),
        *_player_command_args("player-a", config.player_a, config),
        *_player_command_args("player-b", config.player_b, config),
        "--games",
        str(config.games),
        "--start-position-set",
        "random-legal-opening",
        "--start-position-count",
        str(config.start_position_count),
        "--start-position-seed",
        str(config.start_position_seed),
        "--opening-plies",
        str(config.opening_plies),
        "--max-plies",
        str(config.max_plies),
        "--progress-every-games",
        str(config.progress_every_games),
        "--out",
        str(config.games_jsonl.resolve()),
    ]
    return command


def validate_shogi_playing_strength_eval_config(config: ShogiPlayingStrengthEvalConfig) -> None:
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.games % 2:
        raise ValueError("games must be even so each random opening is evaluated on both sides")
    if config.start_position_seed is None:
        raise ValueError("start_position_seed is required")
    if config.opening_plies <= 0:
        raise ValueError("opening_plies must be positive")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if config.max_plies < STANDARD_SHOGI_MAX_PLIES:
        raise ValueError(f"max_plies must be at least {STANDARD_SHOGI_MAX_PLIES} for playing-strength eval")
    if config.simulations <= 0:
        raise ValueError("simulations must be positive")
    if config.nn_leaf_eval_batch_limit <= 0:
        raise ValueError("nn_leaf_eval_batch_limit must be positive")
    if config.mcts_move_time_limit_sec <= 0.0:
        raise ValueError("mcts_move_time_limit_sec must be positive")
    if config.progress_every_games < 0:
        raise ValueError("progress_every_games must be non-negative")
    _validate_player(config.player_a, role="player_a")
    _validate_player(config.player_b, role="player_b")


def _validate_player(player: ShogiPlayingStrengthPlayer, *, role: str) -> None:
    if player.kind == "checkpoint":
        if player.checkpoint is None:
            raise ValueError(f"{role} checkpoint player requires checkpoint")
        return
    if player.kind == "usi_engine":
        if not player.usi_command:
            raise ValueError(f"{role} usi_engine player requires usi_command")
        return
    raise ValueError(f"{role} kind must be checkpoint or usi_engine")


def _player_command_args(prefix: str, player: ShogiPlayingStrengthPlayer, config: ShogiPlayingStrengthEvalConfig) -> list[str]:
    if player.kind == "checkpoint":
        command = [
            f"--{prefix}-kind",
            "checkpoint",
            f"--{prefix}-checkpoint",
            str(player.checkpoint),
            f"--{prefix}-move-selection-profile",
            "max-visit",
            f"--{prefix}-move-selector",
            "mcts",
            f"--{prefix}-mcts-simulations",
            str(config.simulations),
            f"--{prefix}-mcts-nn-leaf-eval-batch-limit",
            str(config.nn_leaf_eval_batch_limit),
            f"--{prefix}-mcts-move-time-limit-sec",
            str(config.mcts_move_time_limit_sec),
            f"--{prefix}-device",
            config.device,
            f"--{prefix}-board-backend",
            config.board_backend,
        ]
        if player.checkpoint_id is not None:
            command.extend([f"--{prefix}-checkpoint-id", player.checkpoint_id])
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


def _player_payload(player: ShogiPlayingStrengthPlayer) -> dict[str, object]:
    payload = asdict(player)
    if player.checkpoint is not None:
        payload["checkpoint"] = str(player.checkpoint)
    return payload


def _shogi_arena_env(arena_repo: Path) -> dict[str, str]:
    pythonpath_parts = [str(arena_repo / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return os.environ | {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}
