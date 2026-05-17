from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from intrep.problems.shogi_policy_value.generated_data_artifacts import (
    ShogiGeneratedDataTrainingCycleResult,
    ShogiGeneratedDataTrainingLoopResult,
    promoted_generated_data_checkpoint,
)
from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    ShogiPolicyValueDataSelectionSource,
    ShogiPolicyValueTargetConstruction,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    DEFAULT_SHOGI_MAX_PLIES,
    ShogiGeneratedPlayerSpec,
    checkpoint_generated_player,
    run_shogi_generated_games,
    warn_short_max_plies,
)
from intrep.problems.shogi_policy_value.online_replay import (
    DEFAULT_GENERATOR_GATE_GAMES,
    DEFAULT_MIN_REPLAY_SIZE,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION,
    DEFAULT_GENERATOR_GATE_WORKER_PROCESSES,
    DEFAULT_GENERATION_WORKER_PROCESSES,
    DEFAULT_TARGET_SAMPLE_PASSES,
    DEFAULT_TRAINING_BATCH_SIZE,
    ShogiOnlineReplayConfig,
    ShogiOnlineReplayIterationResult,
    ShogiGeneratedExperienceSource,
    ShogiOnlineReplayTrainingBudget,
    ShogiOnlineReplayResult,
    run_shogi_online_replay,
)
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl

__all__ = [
    "DEFAULT_MIN_REPLAY_SIZE",
    "DEFAULT_REPLAY_CAPACITY",
    "DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION",
    "DEFAULT_TARGET_SAMPLE_PASSES",
    "DEFAULT_TRAINING_BATCH_SIZE",
    "DEFAULT_GENERATOR_GATE_GAMES",
    "DEFAULT_GENERATOR_GATE_WORKER_PROCESSES",
    "DEFAULT_GENERATION_WORKER_PROCESSES",
    "DEFAULT_SHOGI_MAX_PLIES",
    "ShogiGeneratedDataTrainingCycleConfig",
    "ShogiGeneratedDataTrainingCycleResult",
    "ShogiGeneratedDataTrainingLoopConfig",
    "ShogiGeneratedDataTrainingLoopResult",
    "ShogiGeneratedPlayerSpec",
    "ShogiOnlineReplayConfig",
    "ShogiOnlineReplayIterationResult",
    "ShogiGeneratedExperienceSource",
    "ShogiOnlineReplayTrainingBudget",
    "ShogiOnlineReplayResult",
    "run_shogi_generated_data_training_cycle",
    "run_shogi_generated_data_training_loop",
    "run_shogi_online_replay",
]


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingCycleConfig:
    checkpoint: Path
    run_dir: Path
    arena_repo: Path = Path("../shogi-arena-agent")
    black_player: ShogiGeneratedPlayerSpec = checkpoint_generated_player("black")
    white_player: ShogiGeneratedPlayerSpec = checkpoint_generated_player("white")
    games: int = 4
    concurrent_games_per_process: int = 1
    generation_progress_every_plies: int = 0
    board_backend: str = "cshogi"
    # Computer-shogi self-play should not end as a short artificial draw; use
    # the WCSC-style 320-ply cap as the default and warn on shorter overrides.
    max_plies: int = DEFAULT_SHOGI_MAX_PLIES
    simulations: int = 16
    evaluation_batch_size: int = 1
    generation_worker_processes: int = 1
    seed: int | None = None
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    allow_nonstandard_loss_weights: bool = False
    device: str = "cpu"
    num_workers: int = 0


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingLoopConfig:
    checkpoint: Path
    run_dir: Path
    cycles: int = 1
    next_checkpoint: str = "best"
    arena_repo: Path = Path("../shogi-arena-agent")
    black_player: ShogiGeneratedPlayerSpec = checkpoint_generated_player("black")
    white_player: ShogiGeneratedPlayerSpec = checkpoint_generated_player("white")
    games: int = 4
    concurrent_games_per_process: int = 1
    generation_progress_every_plies: int = 0
    board_backend: str = "cshogi"
    # Computer-shogi self-play should not end as a short artificial draw; use
    # the WCSC-style 320-ply cap as the default and warn on shorter overrides.
    max_plies: int = DEFAULT_SHOGI_MAX_PLIES
    simulations: int = 16
    evaluation_batch_size: int = 1
    generation_worker_processes: int = 1
    seed: int | None = None
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    allow_nonstandard_loss_weights: bool = False
    device: str = "cpu"
    num_workers: int = 0


def run_shogi_generated_data_training_cycle(
    config: ShogiGeneratedDataTrainingCycleConfig,
) -> ShogiGeneratedDataTrainingCycleResult:
    _validate_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    games_jsonl = run_dir / "generated-games.jsonl"
    train_jsonl = run_dir / "train-games.jsonl"
    eval_jsonl = run_dir / "eval-games.jsonl"
    generation_summary_path = run_dir / "generation-summary.json"
    data_selection_json = run_dir / "data-selection.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best-checkpoint.pt"
    metrics_path = run_dir / "metrics.json"

    run_shogi_generated_games(
        arena_repo=config.arena_repo,
        checkpoint=config.checkpoint,
        black_player=config.black_player,
        white_player=config.white_player,
        out=games_jsonl,
        generation_summary_path=generation_summary_path,
        games=config.games,
        concurrent_games_per_process=config.concurrent_games_per_process,
        generation_progress_every_plies=config.generation_progress_every_plies,
        board_backend=config.board_backend,
        max_plies=config.max_plies,
        simulations=config.simulations,
        evaluation_batch_size=config.evaluation_batch_size,
        generation_worker_processes=config.generation_worker_processes,
        seed=config.seed,
        checkpoint_device=config.device,
        mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
    )
    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=games_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        eval_ratio=config.eval_ratio,
    )
    _write_data_selection(data_selection_json, train_jsonl=train_jsonl, eval_jsonl=eval_jsonl)
    _run_training(
        data_selection_json=data_selection_json,
        init_checkpoint_path=config.checkpoint,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        allow_nonstandard_loss_weights=config.allow_nonstandard_loss_weights,
        device=config.device,
        num_workers=config.num_workers,
    )
    generation = {
        "black_player": _player_summary(config.black_player),
        "white_player": _player_summary(config.white_player),
        "games": config.games,
        "concurrent_games_per_process": config.concurrent_games_per_process,
        "generation_progress_every_plies": config.generation_progress_every_plies,
        "board_backend": config.board_backend,
        "max_plies": config.max_plies,
        "simulations": config.simulations,
        "evaluation_batch_size": config.evaluation_batch_size,
        "generation_worker_processes": config.generation_worker_processes,
        "seed": config.seed,
        "checkpoint_device": config.device,
        "mcts_move_time_limit_sec": config.mcts_move_time_limit_sec,
    }
    return ShogiGeneratedDataTrainingCycleResult(
        run_dir=run_dir,
        generated_games_jsonl=games_jsonl,
        train_games=train_count,
        eval_games=eval_count,
        data_selection=data_selection_json,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        metrics=metrics_path,
        generation=generation,
    )


def run_shogi_generated_data_training_loop(
    config: ShogiGeneratedDataTrainingLoopConfig,
) -> ShogiGeneratedDataTrainingLoopResult:
    _validate_loop_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.checkpoint
    results: list[ShogiGeneratedDataTrainingCycleResult] = []
    for cycle_index in range(1, config.cycles + 1):
        result = run_shogi_generated_data_training_cycle(
            ShogiGeneratedDataTrainingCycleConfig(
                checkpoint=checkpoint,
                run_dir=run_dir / f"cycle-{cycle_index:04d}",
                arena_repo=config.arena_repo,
                black_player=config.black_player,
                white_player=config.white_player,
                games=config.games,
                concurrent_games_per_process=config.concurrent_games_per_process,
                generation_progress_every_plies=config.generation_progress_every_plies,
                board_backend=config.board_backend,
                max_plies=config.max_plies,
                simulations=config.simulations,
                evaluation_batch_size=config.evaluation_batch_size,
                generation_worker_processes=config.generation_worker_processes,
                seed=config.seed,
                mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
                eval_ratio=config.eval_ratio,
                max_steps=config.max_steps,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                policy_loss_weight=config.policy_loss_weight,
                value_loss_weight=config.value_loss_weight,
                allow_nonstandard_loss_weights=config.allow_nonstandard_loss_weights,
                device=config.device,
                num_workers=config.num_workers,
            )
        )
        results.append(result)
        checkpoint = promoted_generated_data_checkpoint(result, policy=config.next_checkpoint)
    return ShogiGeneratedDataTrainingLoopResult(
        run_dir=run_dir,
        initial_checkpoint=config.checkpoint,
        final_checkpoint=checkpoint,
        next_checkpoint=config.next_checkpoint,
        cycles=tuple(results),
    )


def _validate_config(config: ShogiGeneratedDataTrainingCycleConfig) -> None:
    _validate_generated_player(config.black_player, side="black")
    _validate_generated_player(config.white_player, side="white")
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.concurrent_games_per_process <= 0:
        raise ValueError("concurrent_games_per_process must be positive")
    if config.generation_progress_every_plies < 0:
        raise ValueError("generation_progress_every_plies must be non-negative")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    warn_short_max_plies(config.max_plies)
    if config.simulations <= 0:
        raise ValueError("simulations must be positive")
    if config.evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
    if config.generation_worker_processes <= 0:
        raise ValueError("generation_worker_processes must be positive")
    if config.mcts_move_time_limit_sec is not None and config.mcts_move_time_limit_sec <= 0.0:
        raise ValueError("mcts_move_time_limit_sec must be positive")
    if not 0.0 < config.eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.policy_loss_weight < 0.0:
        raise ValueError("policy_loss_weight must be non-negative")
    if config.value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be non-negative")
    if config.policy_loss_weight == 0.0 and config.value_loss_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    if not config.allow_nonstandard_loss_weights and (
        config.policy_loss_weight != 1.0 or config.value_loss_weight != 1.0
    ):
        raise ValueError(
            "policy_loss_weight and value_loss_weight default to 1.0; "
            "set allow_nonstandard_loss_weights=True to use other values"
        )
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def _validate_loop_config(config: ShogiGeneratedDataTrainingLoopConfig) -> None:
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    if config.next_checkpoint not in {"best", "final"}:
        raise ValueError("next_checkpoint must be best or final")
    _validate_config(
        ShogiGeneratedDataTrainingCycleConfig(
            checkpoint=config.checkpoint,
            run_dir=config.run_dir,
            arena_repo=config.arena_repo,
            black_player=config.black_player,
            white_player=config.white_player,
            games=config.games,
            concurrent_games_per_process=config.concurrent_games_per_process,
            generation_progress_every_plies=config.generation_progress_every_plies,
            board_backend=config.board_backend,
            max_plies=config.max_plies,
            simulations=config.simulations,
            evaluation_batch_size=config.evaluation_batch_size,
            generation_worker_processes=config.generation_worker_processes,
            seed=config.seed,
            mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
            eval_ratio=config.eval_ratio,
            max_steps=config.max_steps,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            policy_loss_weight=config.policy_loss_weight,
            value_loss_weight=config.value_loss_weight,
            device=config.device,
            num_workers=config.num_workers,
        )
    )


def _validate_generated_player(player: ShogiGeneratedPlayerSpec, *, side: str) -> None:
    if player.kind == "checkpoint":
        return
    if player.kind == "usi_engine":
        if not player.usi_command:
            raise ValueError(f"{side} usi_engine player requires usi_command")
        return
    raise ValueError(f"{side} player kind must be checkpoint or usi_engine")


def _player_summary(player: ShogiGeneratedPlayerSpec) -> dict[str, object]:
    summary: dict[str, object] = {
        "kind": player.kind,
        "name": player.name,
    }
    if player.kind == "checkpoint":
        summary.update(
            {
                "move_selection_profile": player.move_selection_profile,
                "move_selection_temperature": player.move_selection_temperature,
                "move_selection_temperature_plies": player.move_selection_temperature_plies,
            }
        )
    if player.kind == "usi_engine":
        summary.update(
            {
                "usi_options": player.usi_options,
                "usi_go_command": player.usi_go_command,
                "usi_read_timeout_seconds": player.usi_read_timeout_seconds,
            }
        )
    return summary


def _write_data_selection(path: Path, *, train_jsonl: Path, eval_jsonl: Path) -> None:
    selection = ShogiPolicyValueDataSelection(
        name=path.parent.name,
        objective="shogi policy/value from generated game records",
        target_construction=ShogiPolicyValueTargetConstruction(
            policy="chosen_move",
            policy_temperature_cp=100.0,
            policy_mate_cp=100000.0,
            value="winner",
            score_cp_scale=600.0,
        ),
        analysis_sources=(),
        train_sources=(ShogiPolicyValueDataSelectionSource(kind="game_records_jsonl", path=train_jsonl),),
        eval_sources=(ShogiPolicyValueDataSelectionSource(kind="game_records_jsonl", path=eval_jsonl),),
    )
    payload = shogi_policy_value_data_selection_to_json(selection)
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
    allow_nonstandard_loss_weights: bool,
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
    if allow_nonstandard_loss_weights:
        command.append("--allow-nonstandard-loss-weights")
    subprocess.run(command, check=True)
