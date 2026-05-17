from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from intrep.learning.replay_buffer import ReplayBuffer
from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint_training_config,
    load_shogi_policy_value_checkpoint_state_dict,
    save_shogi_policy_value_checkpoint,
    save_shogi_policy_value_state_checkpoint,
)
from intrep.problems.shogi_policy_value.data import load_shogi_policy_value_examples_from_game_records_jsonl
from intrep.problems.shogi_policy_value.data_selection import (
    load_shogi_policy_value_data_selection,
    load_shogi_policy_value_data_selection_examples,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyValueExample,
    TensorizedShogiPolicyValueSample,
    tensorize_shogi_policy_value_examples,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    DEFAULT_SHOGI_MAX_PLIES,
    ShogiGeneratedPlayerSpec,
    checkpoint_generated_player,
    run_shogi_generated_games,
    warn_short_max_plies,
)
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingProgress,
    ShogiPolicyValueTrainingResult,
    train_shogi_policy_value_model,
    validate_shogi_policy_value_loss_weights,
)
from intrep.worlds.shogi.experience_store import append_shogi_experience_store

DEFAULT_REPLAY_CAPACITY = 32768
DEFAULT_SAMPLED_EXAMPLES_PER_CYCLE = 4096
DEFAULT_MIN_REPLAY_SIZE = 8192
DEFAULT_TARGET_SAMPLE_PASSES = 1.0
DEFAULT_TRAINING_BATCH_SIZE = 128
DEFAULT_GENERATOR_GATE_GAMES = 16


@dataclass(frozen=True)
class ShogiGeneratedExperienceSource:
    name: str
    games: int
    black_player: ShogiGeneratedPlayerSpec
    white_player: ShogiGeneratedPlayerSpec


@dataclass(frozen=True)
class ShogiOnlineReplayTrainingBudget:
    sampled_examples_per_cycle: int = DEFAULT_SAMPLED_EXAMPLES_PER_CYCLE
    batch_size: int = DEFAULT_TRAINING_BATCH_SIZE
    target_sample_passes: float = DEFAULT_TARGET_SAMPLE_PASSES
    max_optimizer_steps: int | None = None

    def optimizer_steps_for(self, sampled_examples: int) -> int:
        requested_steps = math.ceil(sampled_examples * self.target_sample_passes / self.batch_size)
        if self.max_optimizer_steps is None:
            return requested_steps
        return min(requested_steps, self.max_optimizer_steps)


@dataclass(frozen=True)
class ShogiOnlineReplayConfig:
    checkpoint: Path
    run_dir: Path
    training_eval_data_selection: Path
    cycles: int = 1
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    min_replay_size: int = DEFAULT_MIN_REPLAY_SIZE
    training_budget: ShogiOnlineReplayTrainingBudget = ShogiOnlineReplayTrainingBudget()
    generator_gate_games: int = DEFAULT_GENERATOR_GATE_GAMES
    experience_store_dir: Path | None = None
    replay_seed_data_selection: Path | None = None
    next_checkpoint: str = "best"
    arena_repo: Path = Path("../shogi-arena-agent")
    experience_sources: tuple[ShogiGeneratedExperienceSource, ...] = (
        ShogiGeneratedExperienceSource(
            name="self-play",
            games=4,
            black_player=checkpoint_generated_player("black"),
            white_player=checkpoint_generated_player("white"),
        ),
    )
    concurrent_games_per_process: int = 1
    generation_progress_every_plies: int = 0
    board_backend: str = "cshogi"
    max_plies: int = DEFAULT_SHOGI_MAX_PLIES
    simulations: int = 128
    evaluation_batch_size: int = 64
    generation_worker_processes: int = 1
    mcts_move_time_limit_sec: float | None = None
    training_config: ShogiPolicyValueTrainingConfig = ShogiPolicyValueTrainingConfig()
    seed: int = 7


@dataclass(frozen=True)
class ShogiOnlineReplayCycleResult:
    cycle_index: int
    run_dir: Path
    generated_games_jsonl: Path
    appended_examples: int
    replay_size: int
    sampled_examples: int
    training_skipped: bool
    experience_store_append: dict[str, object] | None
    checkpoint: Path
    best_checkpoint: Path
    metrics: Path

    def to_json(self) -> dict[str, object]:
        return {
            "cycle_index": self.cycle_index,
            "run_dir": str(self.run_dir),
            "generated_games_jsonl": str(self.generated_games_jsonl),
            "appended_examples": self.appended_examples,
            "replay_size": self.replay_size,
            "sampled_examples": self.sampled_examples,
            "training_skipped": self.training_skipped,
            "experience_store_append": self.experience_store_append,
            "checkpoint": str(self.checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "metrics": str(self.metrics),
        }


@dataclass(frozen=True)
class ShogiOnlineReplayResult:
    run_dir: Path
    initial_checkpoint: Path
    final_checkpoint: Path
    next_checkpoint: str
    replay_capacity: int
    experience_store_dir: Path | None
    replay_seed_data_selection: Path | None
    training_eval_data_selection: Path
    preloaded_examples: int
    training_eval_examples: int
    stop_reason: str | None
    stopped_cycle_index: int | None
    cycles: tuple[ShogiOnlineReplayCycleResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "final_checkpoint": str(self.final_checkpoint),
            "next_checkpoint": self.next_checkpoint,
            "replay_capacity": self.replay_capacity,
            "experience_store_dir": str(self.experience_store_dir) if self.experience_store_dir is not None else None,
            "replay_seed_data_selection": str(self.replay_seed_data_selection) if self.replay_seed_data_selection is not None else None,
            "training_eval_data_selection": str(self.training_eval_data_selection),
            "preloaded_examples": self.preloaded_examples,
            "training_eval_examples": self.training_eval_examples,
            "stop_reason": self.stop_reason,
            "stopped_cycle_index": self.stopped_cycle_index,
            "cycles": [cycle.to_json() for cycle in self.cycles],
        }


@dataclass(frozen=True)
class ShogiOnlineReplayCycleArtifacts:
    cycle_dir: Path
    games_jsonl: Path
    generated_train_jsonl: Path
    generation_summary_path: Path
    generator_gate_games_jsonl: Path
    generator_gate_summary_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path


def run_shogi_online_replay(
    config: ShogiOnlineReplayConfig,
) -> ShogiOnlineReplayResult:
    _validate_online_replay_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.checkpoint
    replay = ReplayBuffer[TensorizedShogiPolicyValueSample](capacity=config.replay_capacity)
    preloaded_examples = _load_replay_seed_examples(config.replay_seed_data_selection)
    training_eval_examples = _load_training_eval_examples(config.training_eval_data_selection)
    training_eval_samples = tensorize_shogi_policy_value_examples(training_eval_examples)
    replay.extend(tensorize_shogi_policy_value_examples(preloaded_examples))
    generator = torch.Generator().manual_seed(config.seed)
    cycle_results: list[ShogiOnlineReplayCycleResult] = []
    last_generator_checkpoint = checkpoint
    stop_reason: str | None = None
    stopped_cycle_index: int | None = None
    for cycle_index in range(1, config.cycles + 1):
        artifacts = _online_replay_cycle_artifacts(run_dir, cycle_index)
        if cycle_index > 1:
            gate_summary = _evaluate_generator_candidate(
                config=config,
                artifacts=artifacts,
                candidate_checkpoint=checkpoint,
                last_generator_checkpoint=last_generator_checkpoint,
            )
            if _generator_candidate_lost(gate_summary):
                stop_reason = "generator_candidate_lost"
                stopped_cycle_index = cycle_index
                break
        _generate_online_replay_cycle_experience(config=config, checkpoint=checkpoint, artifacts=artifacts)
        last_generator_checkpoint = checkpoint
        experience_store_append = _append_to_experience_store(
            store_dir=config.experience_store_dir,
            games_jsonl=artifacts.games_jsonl,
        )
        new_examples = _load_online_replay_cycle_examples(artifacts=artifacts)
        replay.extend(tensorize_shogi_policy_value_examples(new_examples))
        if len(replay) < config.min_replay_size:
            sampled_examples: list[TensorizedShogiPolicyValueSample] = []
            training_skipped = True
            effective_checkpoint = checkpoint
            effective_best_checkpoint = checkpoint
            training_result = None
            skip_reason = "min_replay_size"
        else:
            sampled_examples = replay.sample(
                min(config.training_budget.sampled_examples_per_cycle, len(replay)),
                generator=generator,
            )
            training_result = _train_online_replay_cycle(
                config=config,
                cycle_index=cycle_index,
                checkpoint=checkpoint,
                artifacts=artifacts,
                sampled_examples=sampled_examples,
                eval_examples=training_eval_samples,
                replay_size=len(replay),
                training_eval_examples=len(training_eval_samples),
            )
            training_skipped = False
            effective_checkpoint = artifacts.checkpoint_path
            effective_best_checkpoint = artifacts.best_checkpoint_path
            skip_reason = None
        metrics = _online_replay_cycle_metrics(
            config=config,
            artifacts=artifacts,
            cycle_index=cycle_index,
            training_skipped=training_skipped,
            skip_reason=skip_reason,
            appended_examples=len(new_examples),
            replay_size=len(replay),
            preloaded_examples=len(preloaded_examples),
            training_eval_examples=len(training_eval_examples),
            experience_store_append=experience_store_append,
            sampled_examples=len(sampled_examples),
            init_checkpoint=checkpoint,
            checkpoint=effective_checkpoint,
            best_checkpoint=effective_best_checkpoint,
            training_result=training_result,
        )
        artifacts.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cycle_result = ShogiOnlineReplayCycleResult(
            cycle_index=cycle_index,
            run_dir=artifacts.cycle_dir,
            generated_games_jsonl=artifacts.games_jsonl,
            appended_examples=len(new_examples),
            replay_size=len(replay),
            sampled_examples=len(sampled_examples),
            training_skipped=training_skipped,
            experience_store_append=experience_store_append,
            checkpoint=effective_checkpoint,
            best_checkpoint=effective_best_checkpoint,
            metrics=artifacts.metrics_path,
        )
        cycle_results.append(cycle_result)
        checkpoint = promoted_online_replay_checkpoint(cycle_result, policy=config.next_checkpoint)
    return ShogiOnlineReplayResult(
        run_dir=run_dir,
        initial_checkpoint=config.checkpoint,
        final_checkpoint=checkpoint,
        next_checkpoint=config.next_checkpoint,
        replay_capacity=config.replay_capacity,
        experience_store_dir=config.experience_store_dir,
        replay_seed_data_selection=config.replay_seed_data_selection,
        training_eval_data_selection=config.training_eval_data_selection,
        preloaded_examples=len(preloaded_examples),
        training_eval_examples=len(training_eval_examples),
        stop_reason=stop_reason,
        stopped_cycle_index=stopped_cycle_index,
        cycles=tuple(cycle_results),
    )


def validate_online_replay_config(config: ShogiOnlineReplayConfig) -> None:
    _validate_online_replay_config(config)


def promoted_online_replay_checkpoint(result: ShogiOnlineReplayCycleResult, *, policy: str) -> Path:
    if policy == "best":
        return result.best_checkpoint
    if policy == "final":
        return result.checkpoint
    raise ValueError("next_checkpoint must be best or final")


def _online_replay_cycle_artifacts(run_dir: Path, cycle_index: int) -> ShogiOnlineReplayCycleArtifacts:
    cycle_dir = run_dir / f"cycle-{cycle_index:04d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    return ShogiOnlineReplayCycleArtifacts(
        cycle_dir=cycle_dir,
        games_jsonl=cycle_dir / "generated-games.jsonl",
        generated_train_jsonl=cycle_dir / "generated-train-games.jsonl",
        generation_summary_path=cycle_dir / "generation-summary.json",
        generator_gate_games_jsonl=cycle_dir / "generator-gate-games.jsonl",
        generator_gate_summary_path=cycle_dir / "generator-gate-summary.json",
        checkpoint_path=cycle_dir / "checkpoint.pt",
        best_checkpoint_path=cycle_dir / "best-checkpoint.pt",
        metrics_path=cycle_dir / "metrics.json",
    )


def _generate_online_replay_cycle_experience(
    *,
    config: ShogiOnlineReplayConfig,
    checkpoint: Path,
    artifacts: ShogiOnlineReplayCycleArtifacts,
) -> None:
    source_summaries: list[dict[str, object]] = []
    source_game_paths: list[Path] = []
    for source_index, source in enumerate(config.experience_sources):
        source_dir = artifacts.cycle_dir / f"source-{source_index:03d}-{source.name}"
        source_dir.mkdir(parents=True, exist_ok=True)
        games_jsonl = source_dir / "generated-games.jsonl"
        summary_path = source_dir / "generation-summary.json"
        run_shogi_generated_games(
            arena_repo=config.arena_repo,
            checkpoint=checkpoint,
            black_player=source.black_player,
            white_player=source.white_player,
            out=games_jsonl,
            generation_summary_path=summary_path,
            games=source.games,
            concurrent_games_per_process=config.concurrent_games_per_process,
            generation_progress_every_plies=config.generation_progress_every_plies,
            board_backend=config.board_backend,
            max_plies=config.max_plies,
            simulations=config.simulations,
            evaluation_batch_size=config.evaluation_batch_size,
            generation_worker_processes=config.generation_worker_processes,
            seed=_source_seed(config.seed, artifacts.cycle_dir.name, source_index),
            checkpoint_device=config.training_config.device,
            mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
        )
        source_game_paths.append(games_jsonl)
        source_summaries.append(
            {
                "name": source.name,
                "games": source.games,
                "black_player": _player_summary(source.black_player),
                "white_player": _player_summary(source.white_player),
                "path": str(games_jsonl),
                "summary_path": str(summary_path),
                "summary": _load_json_if_exists(summary_path),
            }
        )
    _merge_jsonl(source_game_paths, artifacts.games_jsonl)
    artifacts.generation_summary_path.write_text(
        json.dumps(_combined_generation_summary(source_summaries), indent=2) + "\n",
        encoding="utf-8",
    )


def _evaluate_generator_candidate(
    *,
    config: ShogiOnlineReplayConfig,
    artifacts: ShogiOnlineReplayCycleArtifacts,
    candidate_checkpoint: Path,
    last_generator_checkpoint: Path,
) -> dict[str, object]:
    if candidate_checkpoint.resolve() == last_generator_checkpoint.resolve():
        summary = {
            "skipped": True,
            "reason": "same_checkpoint",
            "candidate_checkpoint": str(candidate_checkpoint),
            "last_generator_checkpoint": str(last_generator_checkpoint),
            "player_a_wins": 0,
            "player_a_losses": 0,
            "draws": 0,
        }
        artifacts.generator_gate_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    command = [
        sys.executable,
        str(config.arena_repo.resolve() / "scripts/evaluate_shogi_players.py"),
        "--player-a-kind",
        "checkpoint",
        "--player-a-checkpoint",
        str(candidate_checkpoint.resolve()),
        "--player-a-checkpoint-id",
        _checkpoint_actor_id(candidate_checkpoint),
        "--player-a-move-selection-profile",
        "evaluation",
        "--player-a-move-selector",
        "mcts",
        "--player-a-mcts-simulations",
        str(config.simulations),
        "--player-a-mcts-evaluation-batch-size",
        str(config.evaluation_batch_size),
        "--player-a-device",
        config.training_config.device,
        "--player-a-board-backend",
        config.board_backend,
        "--player-b-kind",
        "checkpoint",
        "--player-b-checkpoint",
        str(last_generator_checkpoint.resolve()),
        "--player-b-checkpoint-id",
        _checkpoint_actor_id(last_generator_checkpoint),
        "--player-b-move-selection-profile",
        "evaluation",
        "--player-b-move-selector",
        "mcts",
        "--player-b-mcts-simulations",
        str(config.simulations),
        "--player-b-mcts-evaluation-batch-size",
        str(config.evaluation_batch_size),
        "--player-b-device",
        config.training_config.device,
        "--player-b-board-backend",
        config.board_backend,
        "--out",
        str(artifacts.generator_gate_games_jsonl),
        "--games",
        str(config.generator_gate_games),
        "--max-plies",
        str(config.max_plies),
    ]
    completed = subprocess.run(
        command,
        cwd=config.arena_repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        env=_shogi_arena_env(config.arena_repo),
    )
    summary = json.loads(completed.stdout)
    summary.update(
        {
            "candidate_checkpoint": str(candidate_checkpoint),
            "last_generator_checkpoint": str(last_generator_checkpoint),
        }
    )
    artifacts.generator_gate_summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _generator_candidate_lost(summary: dict[str, object]) -> bool:
    return int(summary.get("player_a_losses", 0)) > int(summary.get("player_a_wins", 0))


def _load_online_replay_cycle_examples(
    *,
    artifacts: ShogiOnlineReplayCycleArtifacts,
) -> list[ShogiPolicyValueExample]:
    artifacts.generated_train_jsonl.write_text(artifacts.games_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
    return _load_generated_policy_value_examples(artifacts.generated_train_jsonl)


def _train_online_replay_cycle(
    *,
    config: ShogiOnlineReplayConfig,
    cycle_index: int,
    checkpoint: Path,
    artifacts: ShogiOnlineReplayCycleArtifacts,
    sampled_examples: list[TensorizedShogiPolicyValueSample],
    eval_examples: list[TensorizedShogiPolicyValueSample],
    replay_size: int,
    training_eval_examples: int,
) -> ShogiPolicyValueTrainingResult:
    training_config = _training_config_from_checkpoint(
        checkpoint,
        config,
        optimizer_steps=config.training_budget.optimizer_steps_for(len(sampled_examples)),
        batch_size=config.training_budget.batch_size,
    )
    training_result = train_shogi_policy_value_model(
        sampled_examples,
        eval_examples=eval_examples,
        config=training_config,
        initial_state_dict=load_shogi_policy_value_checkpoint_state_dict(checkpoint, device=training_config.device),
        progress_callback=_online_replay_training_progress_callback(
            cycle_index=cycle_index,
            replay_size=replay_size,
            sampled_examples=len(sampled_examples),
            training_eval_examples=training_eval_examples,
        ),
    )
    save_shogi_policy_value_checkpoint(artifacts.checkpoint_path, training_result)
    if training_result.best_model_state_dict is not None:
        save_shogi_policy_value_state_checkpoint(
            artifacts.best_checkpoint_path,
            training_result.best_model_state_dict,
            training_result.config,
        )
    else:
        save_shogi_policy_value_checkpoint(artifacts.best_checkpoint_path, training_result)
    return training_result


def _online_replay_training_progress_callback(
    *,
    cycle_index: int,
    replay_size: int,
    sampled_examples: int,
    training_eval_examples: int,
) -> Callable[[ShogiPolicyValueTrainingProgress], None]:
    def report(progress: ShogiPolicyValueTrainingProgress) -> None:
        parts = [
            "online_replay_training_progress",
            f"cycle={cycle_index}",
            f"step={progress.step}/{progress.max_steps}",
            f"loss={progress.loss:.6f}",
            f"elapsed_sec={progress.elapsed_seconds:.1f}",
            f"data_wait_sec={progress.data_wait_seconds:.3f}",
            f"forward_backward_sec={progress.forward_backward_seconds:.3f}",
            f"optimizer_sec={progress.optimizer_seconds:.3f}",
            f"replay_size={replay_size}",
            f"sampled_examples={sampled_examples}",
            f"training_eval_examples={training_eval_examples}",
        ]
        if progress.eval_metrics is not None:
            parts.append(f"eval_loss={progress.eval_metrics.loss:.6f}")
        print(" ".join(parts), flush=True)

    return report


def _online_replay_cycle_metrics(
    *,
    config: ShogiOnlineReplayConfig,
    artifacts: ShogiOnlineReplayCycleArtifacts,
    cycle_index: int,
    training_skipped: bool,
    skip_reason: str | None,
    appended_examples: int,
    replay_size: int,
    preloaded_examples: int,
    training_eval_examples: int,
    experience_store_append: dict[str, object] | None,
    sampled_examples: int,
    init_checkpoint: Path,
    checkpoint: Path,
    best_checkpoint: Path,
    training_result: ShogiPolicyValueTrainingResult | None,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "schema_version": "intrep.shogi_online_replay_metrics.v1",
        "cycle_index": cycle_index,
        "training_skipped": training_skipped,
        "appended_examples": appended_examples,
        "replay_size": replay_size,
        "min_replay_size": config.min_replay_size,
        "experience_store_dir": str(config.experience_store_dir) if config.experience_store_dir is not None else None,
        "replay_seed_data_selection": str(config.replay_seed_data_selection) if config.replay_seed_data_selection is not None else None,
        "training_eval_data_selection": str(config.training_eval_data_selection),
        "preloaded_examples": preloaded_examples,
        "training_eval_examples": training_eval_examples,
        "training_eval_source": "fixed_data_selection",
        "generated_holdout_examples": 0,
        "experience_store_append": experience_store_append,
        "generator_gate_summary_path": str(artifacts.generator_gate_summary_path),
        "generator_gate_summary": _load_json_if_exists(artifacts.generator_gate_summary_path),
        "generation_summary_path": str(artifacts.generation_summary_path),
        "generation_summary": _load_json_if_exists(artifacts.generation_summary_path),
        "sampled_examples": sampled_examples,
        "sampled_examples_per_cycle": config.training_budget.sampled_examples_per_cycle,
        "training_batch_size": config.training_budget.batch_size,
        "target_sample_passes": config.training_budget.target_sample_passes,
        "max_optimizer_steps_per_cycle": config.training_budget.max_optimizer_steps,
        "optimizer_steps_per_cycle": (
            training_result.metrics.actual_steps if training_result is not None else None
        ),
        "effective_sample_passes": _effective_sample_passes(training_result, config, sampled_examples),
        "init_checkpoint_path": str(init_checkpoint),
        "checkpoint_path": str(checkpoint),
        "best_checkpoint_path": str(best_checkpoint),
        "config": asdict(training_result.config) if training_result is not None else None,
        "metrics": asdict(training_result.metrics) if training_result is not None else None,
    }
    if skip_reason is not None:
        metrics["skip_reason"] = skip_reason
    return metrics


def _effective_sample_passes(
    training_result: ShogiPolicyValueTrainingResult | None,
    config: ShogiOnlineReplayConfig,
    sampled_examples: int,
) -> float | None:
    if training_result is None or sampled_examples == 0:
        return None
    return training_result.metrics.actual_steps * config.training_budget.batch_size / sampled_examples


def _load_json_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_online_replay_config(config: ShogiOnlineReplayConfig) -> None:
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.min_replay_size <= 0:
        raise ValueError("min_replay_size must be positive")
    if config.min_replay_size > config.replay_capacity:
        raise ValueError("min_replay_size must be less than or equal to replay_capacity")
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    if config.next_checkpoint not in {"best", "final"}:
        raise ValueError("next_checkpoint must be best or final")
    if config.generator_gate_games <= 0:
        raise ValueError("generator_gate_games must be positive")
    if config.training_eval_data_selection is None:
        raise ValueError("training_eval_data_selection is required")
    if not config.experience_sources:
        raise ValueError("experience_sources must not be empty")
    for source in config.experience_sources:
        _validate_experience_source(source)
    _validate_training_budget(config.training_budget)
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
    training_config = config.training_config
    if training_config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if training_config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    validate_shogi_policy_value_loss_weights(training_config)
    if training_config.max_train_eval_examples is not None and training_config.max_train_eval_examples <= 0:
        raise ValueError("max_train_eval_examples must be positive")
    if training_config.max_eval_examples is not None and training_config.max_eval_examples <= 0:
        raise ValueError("max_eval_examples must be positive")
    if training_config.log_every is not None and training_config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if training_config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if training_config.progress_every is not None and training_config.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if training_config.eval_every is not None and training_config.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if training_config.early_stopping_patience is not None and training_config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if training_config.early_stopping_patience is not None and training_config.eval_every is None:
        raise ValueError("eval_every is required when early_stopping_patience is set")


def _validate_training_budget(training_budget: ShogiOnlineReplayTrainingBudget) -> None:
    if training_budget.sampled_examples_per_cycle <= 0:
        raise ValueError("sampled_examples_per_cycle must be positive")
    if training_budget.batch_size <= 0:
        raise ValueError("training budget batch_size must be positive")
    if training_budget.target_sample_passes <= 0.0:
        raise ValueError("target_sample_passes must be positive")
    if training_budget.max_optimizer_steps is not None and training_budget.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")


def _validate_experience_source(source: ShogiGeneratedExperienceSource) -> None:
    if not source.name:
        raise ValueError("experience source name must not be empty")
    if not all(character.isalnum() or character in "-_" for character in source.name):
        raise ValueError("experience source name must contain only letters, numbers, hyphen, or underscore")
    if source.games <= 0:
        raise ValueError("experience source games must be positive")
    _validate_generated_player(source.black_player, side="black")
    _validate_generated_player(source.white_player, side="white")


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


def _source_seed(base_seed: int, cycle_dir_name: str, source_index: int) -> int:
    cycle_index = int(cycle_dir_name.removeprefix("cycle-"))
    return base_seed + (cycle_index - 1) * 10000 + source_index


def _merge_jsonl(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as merged:
        for path in inputs:
            merged.write(path.read_text(encoding="utf-8"))


def _checkpoint_actor_id(checkpoint: Path) -> str:
    if checkpoint.name in {"checkpoint.pt", "best-checkpoint.pt"}:
        return checkpoint.parent.name
    return checkpoint.stem


def _shogi_arena_env(arena_repo: Path) -> dict[str, str]:
    pythonpath_parts = [str(arena_repo.resolve() / "src")]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return os.environ | {"PYTHONPATH": os.pathsep.join(pythonpath_parts)}


def _combined_generation_summary(source_summaries: list[dict[str, object]]) -> dict[str, object]:
    game_count = 0
    total_plies = 0.0
    wall_time = 0.0
    end_reasons: dict[str, int] = {}
    black_wins = 0
    white_wins = 0
    draws = 0
    for source in source_summaries:
        summary = source.get("summary")
        if not isinstance(summary, dict):
            continue
        source_games = int(summary.get("game_count", 0))
        game_count += source_games
        total_plies += float(summary.get("average_plies", 0.0)) * source_games
        wall_time += float(summary.get("generation_wall_time_sec", 0.0))
        black_wins += int(summary.get("black_wins", 0))
        white_wins += int(summary.get("white_wins", 0))
        draws += int(summary.get("draws", 0))
        for reason, count in dict(summary.get("end_reasons", {})).items():
            end_reasons[str(reason)] = end_reasons.get(str(reason), 0) + int(count)
    return {
        "game_count": game_count,
        "end_reasons": end_reasons,
        "average_plies": total_plies / game_count if game_count else 0.0,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "generation_wall_time_sec": wall_time,
        "plies_per_sec": total_plies / wall_time if wall_time > 0.0 else 0.0,
        "sources": source_summaries,
    }


def _load_replay_seed_examples(data_selection_path: Path | None) -> list[ShogiPolicyValueExample]:
    if data_selection_path is None:
        return []
    selection = load_shogi_policy_value_data_selection(data_selection_path)
    train_examples, _eval_examples = load_shogi_policy_value_data_selection_examples(selection)
    return train_examples


def _load_training_eval_examples(data_selection_path: Path) -> list[ShogiPolicyValueExample]:
    selection = load_shogi_policy_value_data_selection(data_selection_path)
    _train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(selection)
    return eval_examples


def _append_to_experience_store(*, store_dir: Path | None, games_jsonl: Path) -> dict[str, object] | None:
    if store_dir is None:
        return None
    return append_shogi_experience_store(input_path=games_jsonl, store_dir=store_dir)


def _load_generated_policy_value_examples(path: Path) -> list[ShogiPolicyValueExample]:
    return load_shogi_policy_value_examples_from_game_records_jsonl(
        path,
        policy_target_construction="chosen_move",
        value_target_construction="winner",
        policy_temperature_cp=100.0,
        policy_mate_cp=100000.0,
        score_cp_scale=600.0,
    )


def _training_config_from_checkpoint(
    checkpoint: Path,
    config: ShogiOnlineReplayConfig,
    *,
    optimizer_steps: int,
    batch_size: int,
) -> ShogiPolicyValueTrainingConfig:
    training_config = config.training_config
    checkpoint_config = load_shogi_policy_value_checkpoint_training_config(checkpoint, device=training_config.device)
    return ShogiPolicyValueTrainingConfig(
        max_steps=optimizer_steps,
        batch_size=batch_size,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        seed=training_config.seed,
        embedding_dim=checkpoint_config.embedding_dim,
        hidden_dim=checkpoint_config.hidden_dim,
        num_heads=checkpoint_config.num_heads,
        num_layers=checkpoint_config.num_layers,
        use_shared_core=checkpoint_config.use_shared_core,
        policy_loss_weight=training_config.policy_loss_weight,
        value_loss_weight=training_config.value_loss_weight,
        allow_nonstandard_loss_weights=training_config.allow_nonstandard_loss_weights,
        device=training_config.device,
        max_train_eval_examples=training_config.max_train_eval_examples,
        max_eval_examples=training_config.max_eval_examples,
        log_every=training_config.log_every,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        progress_every=training_config.progress_every,
        eval_every=training_config.eval_every,
        early_stopping_patience=training_config.early_stopping_patience,
    )
