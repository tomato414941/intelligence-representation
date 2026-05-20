from __future__ import annotations

import json
import gc
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from intrep.learning.replay_buffer import ReplayBuffer
from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint_identity,
    load_shogi_policy_value_checkpoint_training_config,
    load_shogi_policy_value_checkpoint_state_dict,
    save_shogi_policy_value_checkpoint,
    save_shogi_policy_value_state_checkpoint,
)
from intrep.problems.shogi_policy_value.data import (
    load_shogi_move_policy_value_examples_from_game_records_jsonl,
)
from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    load_shogi_policy_value_data_selection,
    load_shogi_policy_value_data_selection_examples,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiMovePolicyValueExample,
    LegalMovePolicyValueTensorSample,
    load_shogi_move_policy_value_examples_jsonl,
    tensorize_legal_move_policy_value_examples,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    DEFAULT_SHOGI_MAX_PLIES,
    ShogiGeneratedPlayerSpec,
    checkpoint_generated_player,
    run_shogi_generated_games,
    warn_short_max_plies,
)
from intrep.problems.shogi_policy_value.tensor_cache import (
    default_shogi_policy_value_tensor_cache_path,
    load_shogi_policy_value_tensor_cache,
)
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingProgress,
    ShogiPolicyValueTrainingResult,
    train_shogi_policy_value_model,
    validate_shogi_policy_value_loss_weights,
)
from intrep.worlds.shogi.game_record import ShogiGameRecord, load_shogi_game_records_jsonl

DEFAULT_REPLAY_CAPACITY = 2_097_152
DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION = 524_288
DEFAULT_MAX_SEED_EXAMPLES_PER_ITERATION = 50_000
DEFAULT_MIN_REPLAY_SIZE = 8192
DEFAULT_TARGET_SAMPLE_PASSES = 1.0
DEFAULT_TRAINING_BATCH_SIZE = 128
DEFAULT_GENERATOR_GATE_GAMES = 32
DEFAULT_GENERATOR_GATE_WORKER_PROCESSES = 4
DEFAULT_GENERATION_WORKER_PROCESSES = 8


@dataclass(frozen=True)
class ShogiGeneratedExperienceSource:
    name: str
    games: int
    black_player: ShogiGeneratedPlayerSpec
    white_player: ShogiGeneratedPlayerSpec
    policy_target_construction: str = "mcts_visit_counts"
    value_target_construction: str = "winner"


@dataclass(frozen=True)
class ShogiOnlineReplayTrainingBudget:
    sampled_examples_per_iteration: int = DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION
    max_seed_examples_per_iteration: int = DEFAULT_MAX_SEED_EXAMPLES_PER_ITERATION
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
    iterations: int = 1
    resume: bool = False
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    min_replay_size: int = DEFAULT_MIN_REPLAY_SIZE
    training_budget: ShogiOnlineReplayTrainingBudget = ShogiOnlineReplayTrainingBudget()
    generator_gate_games: int = DEFAULT_GENERATOR_GATE_GAMES
    generator_gate_worker_processes: int = DEFAULT_GENERATOR_GATE_WORKER_PROCESSES
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
    nn_leaf_eval_batch_limit: int = 64
    generation_worker_processes: int = DEFAULT_GENERATION_WORKER_PROCESSES
    mcts_move_time_limit_sec: float | None = None
    training_config: ShogiPolicyValueTrainingConfig = ShogiPolicyValueTrainingConfig()
    seed: int = 7


@dataclass(frozen=True)
class ShogiOnlineReplayIterationResult:
    iteration_index: int
    run_dir: Path
    generated_games_jsonl: Path
    appended_examples: int
    replay_size: int
    sampled_examples: int
    training_skipped: bool
    checkpoint: Path
    best_checkpoint: Path
    metrics: Path

    def to_json(self) -> dict[str, object]:
        return {
            "iteration_index": self.iteration_index,
            "run_dir": str(self.run_dir),
            "generated_games_jsonl": str(self.generated_games_jsonl),
            "appended_examples": self.appended_examples,
            "replay_size": self.replay_size,
            "sampled_examples": self.sampled_examples,
            "training_skipped": self.training_skipped,
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
    replay_seed_data_selection: Path | None
    training_eval_data_selection: Path
    preloaded_examples: int
    training_eval_examples: int
    stop_reason: str | None
    stopped_iteration_index: int | None
    iterations: tuple[ShogiOnlineReplayIterationResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "final_checkpoint": str(self.final_checkpoint),
            "next_checkpoint": self.next_checkpoint,
            "replay_capacity": self.replay_capacity,
            "replay_seed_data_selection": str(self.replay_seed_data_selection) if self.replay_seed_data_selection is not None else None,
            "training_eval_data_selection": str(self.training_eval_data_selection),
            "preloaded_examples": self.preloaded_examples,
            "training_eval_examples": self.training_eval_examples,
            "stop_reason": self.stop_reason,
            "stopped_iteration_index": self.stopped_iteration_index,
            "iterations": [iteration.to_json() for iteration in self.iterations],
        }


@dataclass(frozen=True)
class ShogiOnlineReplayIterationArtifacts:
    iteration_dir: Path
    games_jsonl: Path
    generation_summary_path: Path
    generator_gate_games_jsonl: Path
    generator_gate_result_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class _OnlineReplayResumeState:
    generated_replay: ReplayBuffer[LegalMovePolicyValueTensorSample]
    iteration_results: tuple[ShogiOnlineReplayIterationResult, ...]
    checkpoint: Path
    last_generator_checkpoint: Path
    next_iteration_index: int


def run_shogi_online_replay(
    config: ShogiOnlineReplayConfig,
) -> ShogiOnlineReplayResult:
    _validate_online_replay_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    replay_seed_selection = _load_replay_seed_selection(config.replay_seed_data_selection)
    replay_seed_eligible_examples = _count_replay_seed_examples(replay_seed_selection)
    training_eval_samples = _load_training_eval_samples(config.training_eval_data_selection)
    generator = torch.Generator().manual_seed(config.seed)
    if config.resume:
        resume_state = _load_online_replay_resume_state(config=config, run_dir=run_dir, generator=generator)
        generated_replay = resume_state.generated_replay
        iteration_results = list(resume_state.iteration_results)
        checkpoint = resume_state.checkpoint
        last_generator_checkpoint = resume_state.last_generator_checkpoint
        start_iteration_index = resume_state.next_iteration_index
    else:
        generated_replay = ReplayBuffer[LegalMovePolicyValueTensorSample](capacity=config.replay_capacity)
        iteration_results = []
        checkpoint = config.checkpoint
        last_generator_checkpoint = checkpoint
        start_iteration_index = 1
    stop_reason: str | None = None
    stopped_iteration_index: int | None = None
    for iteration_index in range(start_iteration_index, config.iterations + 1):
        iteration_start = time.perf_counter()
        phase_timings: dict[str, float | None] = {
            "gate_wall_time_sec": None,
            "generation_wall_time_sec": None,
            "generated_train_extraction_wall_time_sec": None,
            "generated_tensorize_wall_time_sec": None,
            "replay_sampling_wall_time_sec": None,
            "training_wall_time_sec": None,
            "checkpoint_save_wall_time_sec": None,
        }
        artifacts = _prepare_online_replay_iteration_artifacts(run_dir, iteration_index)
        if iteration_index > 1:
            gate_start = time.perf_counter()
            gate_result = _evaluate_generator_candidate(
                config=config,
                artifacts=artifacts,
                candidate_checkpoint=checkpoint,
                last_generator_checkpoint=last_generator_checkpoint,
            )
            phase_timings["gate_wall_time_sec"] = time.perf_counter() - gate_start
            if _generator_gate_should_stop(gate_result):
                stop_reason = "generator_candidate_clearly_worse"
                stopped_iteration_index = iteration_index
                break
        generation_start = time.perf_counter()
        _generate_online_replay_iteration_experience(config=config, checkpoint=checkpoint, artifacts=artifacts)
        phase_timings["generation_wall_time_sec"] = time.perf_counter() - generation_start
        last_generator_checkpoint = checkpoint
        generated_train_extraction_start = time.perf_counter()
        new_examples = _load_online_replay_generated_examples(config=config, artifacts=artifacts)
        phase_timings["generated_train_extraction_wall_time_sec"] = (
            time.perf_counter() - generated_train_extraction_start
        )
        generated_tensorize_start = time.perf_counter()
        generated_replay.extend(tensorize_legal_move_policy_value_examples(new_examples))
        phase_timings["generated_tensorize_wall_time_sec"] = time.perf_counter() - generated_tensorize_start
        replay_size = replay_seed_eligible_examples + len(generated_replay)
        if replay_size < config.min_replay_size:
            sampled_examples: list[LegalMovePolicyValueTensorSample] = []
            seed_sampled_examples = 0
            generated_sampled_examples = 0
            training_skipped = True
            effective_checkpoint = checkpoint
            effective_best_checkpoint = checkpoint
            training_result = None
            skip_reason = "min_replay_size"
        else:
            replay_sampling_start = time.perf_counter()
            generated_sampled_examples = len(generated_replay)
            generated_samples = generated_replay.sample(
                generated_sampled_examples,
                generator=generator,
            )
            seed_sampled_examples = min(
                replay_seed_eligible_examples,
                config.training_budget.max_seed_examples_per_iteration,
                max(0, config.training_budget.sampled_examples_per_iteration - generated_sampled_examples),
            )
            seed_samples = _sample_replay_seed_samples(
                replay_seed_selection,
                sample_count=seed_sampled_examples,
                seed=config.seed + iteration_index,
            )
            sampled_examples = generated_samples + seed_samples
            phase_timings["replay_sampling_wall_time_sec"] = time.perf_counter() - replay_sampling_start
            training_start = time.perf_counter()
            training_result = _train_online_replay_iteration(
                config=config,
                iteration_index=iteration_index,
                checkpoint=checkpoint,
                sampled_examples=sampled_examples,
                eval_examples=training_eval_samples,
                replay_size=replay_size,
                training_eval_examples=len(training_eval_samples),
            )
            phase_timings["training_wall_time_sec"] = time.perf_counter() - training_start
            checkpoint_save_start = time.perf_counter()
            _save_online_replay_iteration_checkpoints(artifacts=artifacts, training_result=training_result)
            phase_timings["checkpoint_save_wall_time_sec"] = time.perf_counter() - checkpoint_save_start
            training_skipped = False
            effective_checkpoint = artifacts.checkpoint_path
            effective_best_checkpoint = artifacts.best_checkpoint_path
            skip_reason = None
        phase_timings["iteration_wall_time_sec"] = time.perf_counter() - iteration_start
        metrics = _online_replay_iteration_metrics(
            config=config,
            artifacts=artifacts,
            iteration_index=iteration_index,
            training_skipped=training_skipped,
            skip_reason=skip_reason,
            appended_examples=len(new_examples),
            replay_size=replay_size,
            generated_replay_size=len(generated_replay),
            replay_seed_eligible_examples=replay_seed_eligible_examples,
            seed_sampled_examples=seed_sampled_examples,
            generated_sampled_examples=generated_sampled_examples,
            training_eval_examples=len(training_eval_samples),
            sampled_examples=len(sampled_examples),
            init_checkpoint=checkpoint,
            checkpoint=effective_checkpoint,
            best_checkpoint=effective_best_checkpoint,
            training_result=training_result,
            phase_timings=phase_timings,
        )
        artifacts.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        training_result = None
        _clear_cuda_cache_if_needed(config.training_config.device)
        iteration_result = ShogiOnlineReplayIterationResult(
            iteration_index=iteration_index,
            run_dir=artifacts.iteration_dir,
            generated_games_jsonl=artifacts.games_jsonl,
            appended_examples=len(new_examples),
            replay_size=replay_size,
            sampled_examples=len(sampled_examples),
            training_skipped=training_skipped,
            checkpoint=effective_checkpoint,
            best_checkpoint=effective_best_checkpoint,
            metrics=artifacts.metrics_path,
        )
        iteration_results.append(iteration_result)
        checkpoint = promoted_online_replay_checkpoint(iteration_result, policy=config.next_checkpoint)
    return ShogiOnlineReplayResult(
        run_dir=run_dir,
        initial_checkpoint=config.checkpoint,
        final_checkpoint=checkpoint,
        next_checkpoint=config.next_checkpoint,
        replay_capacity=config.replay_capacity,
        replay_seed_data_selection=config.replay_seed_data_selection,
        training_eval_data_selection=config.training_eval_data_selection,
        preloaded_examples=0,
        training_eval_examples=len(training_eval_samples),
        stop_reason=stop_reason,
        stopped_iteration_index=stopped_iteration_index,
        iterations=tuple(iteration_results),
    )


def _clear_cuda_cache_if_needed(device: str) -> None:
    if not device.startswith("cuda"):
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_online_replay_config(config: ShogiOnlineReplayConfig) -> None:
    _validate_online_replay_config(config)


def promoted_online_replay_checkpoint(result: ShogiOnlineReplayIterationResult, *, policy: str) -> Path:
    if policy == "best":
        return result.best_checkpoint
    if policy == "final":
        return result.checkpoint
    raise ValueError("next_checkpoint must be best or final")


def _online_replay_iteration_artifacts(run_dir: Path, iteration_index: int) -> ShogiOnlineReplayIterationArtifacts:
    iteration_dir = run_dir / f"iteration-{iteration_index:04d}"
    return ShogiOnlineReplayIterationArtifacts(
        iteration_dir=iteration_dir,
        games_jsonl=iteration_dir / "generated-games.jsonl",
        generation_summary_path=iteration_dir / "generation-summary.json",
        generator_gate_games_jsonl=iteration_dir / "generator-gate-games.jsonl",
        generator_gate_result_path=iteration_dir / "generator-gate-result.json",
        checkpoint_path=iteration_dir / "checkpoint.pt",
        best_checkpoint_path=iteration_dir / "best-checkpoint.pt",
        metrics_path=iteration_dir / "metrics.json",
    )


def _prepare_online_replay_iteration_artifacts(run_dir: Path, iteration_index: int) -> ShogiOnlineReplayIterationArtifacts:
    artifacts = _online_replay_iteration_artifacts(run_dir, iteration_index)
    artifacts.iteration_dir.mkdir(parents=True, exist_ok=True)
    return artifacts


def _load_online_replay_resume_state(
    *,
    config: ShogiOnlineReplayConfig,
    run_dir: Path,
    generator: torch.Generator,
) -> _OnlineReplayResumeState:
    generated_replay = ReplayBuffer[LegalMovePolicyValueTensorSample](capacity=config.replay_capacity)
    iteration_results: list[ShogiOnlineReplayIterationResult] = []
    checkpoint = config.checkpoint
    last_generator_checkpoint = config.checkpoint
    for iteration_index in range(1, config.iterations + 1):
        artifacts = _online_replay_iteration_artifacts(run_dir, iteration_index)
        if not artifacts.metrics_path.exists():
            break
        metrics = _load_completed_online_replay_iteration_metrics(
            config=config,
            artifacts=artifacts,
            iteration_index=iteration_index,
        )
        new_examples = _load_online_replay_generated_examples(config=config, artifacts=artifacts)
        generated_replay.extend(tensorize_legal_move_policy_value_examples(new_examples))
        _advance_online_replay_resume_rng(
            generated_replay=generated_replay,
            metrics=metrics,
            generator=generator,
        )
        iteration_result = _online_replay_iteration_result_from_metrics(
            artifacts=artifacts,
            iteration_index=iteration_index,
            metrics=metrics,
        )
        iteration_results.append(iteration_result)
        checkpoint = promoted_online_replay_checkpoint(iteration_result, policy=config.next_checkpoint)
        last_generator_checkpoint = Path(str(metrics["checkpoint"]["init_path"]))
    return _OnlineReplayResumeState(
        generated_replay=generated_replay,
        iteration_results=tuple(iteration_results),
        checkpoint=checkpoint,
        last_generator_checkpoint=last_generator_checkpoint,
        next_iteration_index=len(iteration_results) + 1,
    )


def _load_completed_online_replay_iteration_metrics(
    *,
    config: ShogiOnlineReplayConfig,
    artifacts: ShogiOnlineReplayIterationArtifacts,
    iteration_index: int,
) -> dict[str, object]:
    metrics = _load_json_if_exists(artifacts.metrics_path)
    if not isinstance(metrics, dict):
        raise ValueError(f"cannot resume from invalid Online Replay metrics: {artifacts.metrics_path}")
    if metrics.get("schema_version") != "intrep.shogi_online_replay_metrics.v1":
        raise ValueError(f"cannot resume from unsupported Online Replay metrics: {artifacts.metrics_path}")
    if int(metrics.get("iteration_index", 0)) != iteration_index:
        raise ValueError(f"cannot resume from mismatched iteration metrics: {artifacts.metrics_path}")
    _validate_online_replay_resume_metrics(config=config, metrics=metrics, iteration_index=iteration_index)
    return metrics


def _validate_online_replay_resume_metrics(
    *,
    config: ShogiOnlineReplayConfig,
    metrics: dict[str, object],
    iteration_index: int,
) -> None:
    replay = dict(metrics.get("replay", {}))
    if int(replay.get("capacity", 0)) != config.replay_capacity:
        raise ValueError("cannot resume Online Replay with a different replay_capacity")
    expected_seed_selection = str(config.replay_seed_data_selection) if config.replay_seed_data_selection is not None else None
    if replay.get("seed_data_selection") != expected_seed_selection:
        raise ValueError("cannot resume Online Replay with a different replay_seed_data_selection")
    checkpoint = dict(metrics.get("checkpoint", {}))
    if iteration_index == 1 and checkpoint.get("init_id") != _checkpoint_actor_id(config.checkpoint):
        raise ValueError("cannot resume Online Replay from a different initial checkpoint")
    gate = dict(metrics.get("gate", {}))
    gate_config = dict(gate.get("config", {}))
    if int(gate_config.get("mcts_simulations", config.simulations)) != config.simulations:
        raise ValueError("cannot resume Online Replay with different MCTS simulations")
    if int(gate_config.get("nn_leaf_eval_batch_limit", config.nn_leaf_eval_batch_limit)) != config.nn_leaf_eval_batch_limit:
        raise ValueError("cannot resume Online Replay with a different NN leaf eval batch limit")
    if int(gate_config.get("max_plies", config.max_plies)) != config.max_plies:
        raise ValueError("cannot resume Online Replay with a different max_plies")
    generation = dict(metrics.get("generation", {}))
    summary = generation.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("cannot resume Online Replay without a generation summary")
    source_payloads = summary.get("sources")
    if not isinstance(source_payloads, list):
        raise ValueError("cannot resume Online Replay without generation source metadata")
    expected_sources = config.experience_sources
    if len(source_payloads) != len(expected_sources):
        raise ValueError("cannot resume Online Replay with different experience sources")
    for source_payload, expected_source in zip(source_payloads, expected_sources):
        source = dict(source_payload)
        if source.get("name") != expected_source.name:
            raise ValueError("cannot resume Online Replay with different experience sources")
        if source.get("policy_target_construction") != expected_source.policy_target_construction:
            raise ValueError("cannot resume Online Replay with different policy target construction")
        if source.get("value_target_construction") != expected_source.value_target_construction:
            raise ValueError("cannot resume Online Replay with different value target construction")


def _advance_online_replay_resume_rng(
    *,
    generated_replay: ReplayBuffer[LegalMovePolicyValueTensorSample],
    metrics: dict[str, object],
    generator: torch.Generator,
) -> None:
    training = dict(metrics.get("training", {}))
    if bool(training.get("skipped", False)):
        return
    replay = dict(metrics.get("replay", {}))
    generated_sampled_examples = int(replay.get("generated_sampled_examples", 0))
    if generated_sampled_examples > 0:
        generated_replay.sample(generated_sampled_examples, generator=generator)


def _online_replay_iteration_result_from_metrics(
    *,
    artifacts: ShogiOnlineReplayIterationArtifacts,
    iteration_index: int,
    metrics: dict[str, object],
) -> ShogiOnlineReplayIterationResult:
    replay = dict(metrics.get("replay", {}))
    generation = dict(metrics.get("generation", {}))
    training = dict(metrics.get("training", {}))
    checkpoint = dict(metrics.get("checkpoint", {}))
    return ShogiOnlineReplayIterationResult(
        iteration_index=iteration_index,
        run_dir=artifacts.iteration_dir,
        generated_games_jsonl=artifacts.games_jsonl,
        appended_examples=int(generation.get("appended_examples", 0)),
        replay_size=int(replay.get("size", 0)),
        sampled_examples=int(replay.get("sampled_examples", 0)),
        training_skipped=bool(training.get("skipped", False)),
        checkpoint=Path(str(checkpoint["path"])),
        best_checkpoint=Path(str(checkpoint["best_path"])),
        metrics=artifacts.metrics_path,
    )


def _generate_online_replay_iteration_experience(
    *,
    config: ShogiOnlineReplayConfig,
    checkpoint: Path,
    artifacts: ShogiOnlineReplayIterationArtifacts,
) -> None:
    source_summaries: list[dict[str, object]] = []
    source_game_paths: list[Path] = []
    for source_index, source in enumerate(config.experience_sources):
        source_dir = artifacts.iteration_dir / f"source-{source_index:03d}-{source.name}"
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
            nn_leaf_eval_batch_limit=config.nn_leaf_eval_batch_limit,
            generation_worker_processes=config.generation_worker_processes,
            seed=_source_seed(config.seed, artifacts.iteration_dir.name, source_index),
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
                "policy_target_construction": source.policy_target_construction,
                "value_target_construction": source.value_target_construction,
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
    artifacts: ShogiOnlineReplayIterationArtifacts,
    candidate_checkpoint: Path,
    last_generator_checkpoint: Path,
) -> dict[str, object]:
    if candidate_checkpoint.resolve() == last_generator_checkpoint.resolve():
        checkpoint_identity = load_shogi_policy_value_checkpoint_identity(candidate_checkpoint)
        result = {
            "skipped": True,
            "reason": "same_checkpoint",
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_checkpoint_id": checkpoint_identity.checkpoint_id,
            "last_generator_checkpoint": str(last_generator_checkpoint),
            "last_generator_checkpoint_id": checkpoint_identity.checkpoint_id,
            "player_a_wins": 0,
            "player_a_losses": 0,
            "draws": 0,
            "match_worker_processes": config.generator_gate_worker_processes,
        }
        result.update(_generator_gate_decision(result))
        result["side_breakdown"] = _empty_generator_gate_side_breakdown()
        artifacts.generator_gate_result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
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
        "visit-sampling",
        "--player-a-move-selector",
        "mcts",
        "--player-a-mcts-simulations",
        str(config.simulations),
        "--player-a-mcts-nn-leaf-eval-batch-limit",
        str(config.nn_leaf_eval_batch_limit),
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
        "visit-sampling",
        "--player-b-move-selector",
        "mcts",
        "--player-b-mcts-simulations",
        str(config.simulations),
        "--player-b-mcts-nn-leaf-eval-batch-limit",
        str(config.nn_leaf_eval_batch_limit),
        "--player-b-device",
        config.training_config.device,
        "--player-b-board-backend",
        config.board_backend,
        "--out",
        str(artifacts.generator_gate_games_jsonl),
        "--games",
        str(config.generator_gate_games),
        "--match-worker-processes",
        str(config.generator_gate_worker_processes),
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
    result = json.loads(completed.stdout)
    result.update(
        {
            "candidate_checkpoint": str(candidate_checkpoint),
            "last_generator_checkpoint": str(last_generator_checkpoint),
        }
    )
    result.update(_generator_gate_decision(result))
    result["side_breakdown"] = _generator_gate_side_breakdown(
        artifacts.generator_gate_games_jsonl,
        player_a_name=_checkpoint_actor_id(candidate_checkpoint),
    )
    artifacts.generator_gate_result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _generator_gate_decision(result: dict[str, object]) -> dict[str, object]:
    wins = int(result.get("player_a_wins", 0))
    losses = int(result.get("player_a_losses", 0))
    draws = int(result.get("draws", 0))
    margin = wins - losses
    if margin <= -2:
        decision = "clearly_worse"
    elif margin >= 2:
        decision = "favorable"
    else:
        decision = "unclear"
    return {
        "decision": decision,
        "should_stop": decision == "clearly_worse",
        "margin": margin,
        "decisive_games": wins + losses,
        "draws": draws,
        "interpretation": "degradation_guard",
    }


def _generator_gate_should_stop(result: dict[str, object]) -> bool:
    return bool(result.get("should_stop", False))


def _empty_generator_gate_side_breakdown() -> dict[str, object]:
    return {
        "policy": "recorded_not_used_for_stopping",
        "player_a_as_black": _empty_gate_side_result(),
        "player_a_as_white": _empty_gate_side_result(),
        "player_a_side_unknown": _empty_gate_side_result(),
    }


def _generator_gate_side_breakdown(games_jsonl: Path, *, player_a_name: str) -> dict[str, object]:
    breakdown = _empty_generator_gate_side_breakdown()
    for record in load_shogi_game_records_jsonl(games_jsonl):
        side_key = _player_a_side_key(record, player_a_name=player_a_name)
        side_result = dict(breakdown[side_key])
        side_result["games"] = int(side_result["games"]) + 1
        if record.winner is None:
            side_result["draws"] = int(side_result["draws"]) + 1
        elif (side_key == "player_a_as_black" and record.winner == "black") or (
            side_key == "player_a_as_white" and record.winner == "white"
        ):
            side_result["wins"] = int(side_result["wins"]) + 1
        elif side_key == "player_a_side_unknown":
            side_result["unknown_results"] = int(side_result["unknown_results"]) + 1
        else:
            side_result["losses"] = int(side_result["losses"]) + 1
        breakdown[side_key] = side_result
    return breakdown


def _empty_gate_side_result() -> dict[str, int]:
    return {
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "unknown_results": 0,
    }


def _player_a_side_key(record: ShogiGameRecord, *, player_a_name: str) -> str:
    if record.black_actor.name == player_a_name:
        return "player_a_as_black"
    if record.white_actor.name == player_a_name:
        return "player_a_as_white"
    return "player_a_side_unknown"


def _load_online_replay_generated_examples(
    *,
    config: ShogiOnlineReplayConfig,
    artifacts: ShogiOnlineReplayIterationArtifacts,
) -> list[ShogiMovePolicyValueExample]:
    examples: list[ShogiMovePolicyValueExample] = []
    for source_index, source in enumerate(config.experience_sources):
        source_path = artifacts.iteration_dir / f"source-{source_index:03d}-{source.name}" / "generated-games.jsonl"
        examples.extend(
            _load_generated_policy_value_examples(
                source_path,
                policy_target_construction=source.policy_target_construction,
                value_target_construction=source.value_target_construction,
            )
        )
    return examples


def _train_online_replay_iteration(
    *,
    config: ShogiOnlineReplayConfig,
    iteration_index: int,
    checkpoint: Path,
    sampled_examples: list[LegalMovePolicyValueTensorSample],
    eval_examples: Sequence[LegalMovePolicyValueTensorSample],
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
            iteration_index=iteration_index,
            replay_size=replay_size,
            sampled_examples=len(sampled_examples),
            training_eval_examples=training_eval_examples,
        ),
    )
    return training_result


def _save_online_replay_iteration_checkpoints(
    *,
    artifacts: ShogiOnlineReplayIterationArtifacts,
    training_result: ShogiPolicyValueTrainingResult,
) -> None:
    save_shogi_policy_value_checkpoint(artifacts.checkpoint_path, training_result)
    if training_result.best_model_state_dict is not None:
        save_shogi_policy_value_state_checkpoint(
            artifacts.best_checkpoint_path,
            training_result.best_model_state_dict,
            training_result.config,
        )
    else:
        save_shogi_policy_value_checkpoint(artifacts.best_checkpoint_path, training_result)


def _online_replay_training_progress_callback(
    *,
    iteration_index: int,
    replay_size: int,
    sampled_examples: int,
    training_eval_examples: int,
) -> Callable[[ShogiPolicyValueTrainingProgress], None]:
    def report(progress: ShogiPolicyValueTrainingProgress) -> None:
        parts = [
            "online_replay_training_progress",
            f"iteration={iteration_index}",
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


def _online_replay_iteration_metrics(
    *,
    config: ShogiOnlineReplayConfig,
    artifacts: ShogiOnlineReplayIterationArtifacts,
    iteration_index: int,
    training_skipped: bool,
    skip_reason: str | None,
    appended_examples: int,
    replay_size: int,
    generated_replay_size: int,
    replay_seed_eligible_examples: int,
    seed_sampled_examples: int,
    generated_sampled_examples: int,
    training_eval_examples: int,
    sampled_examples: int,
    init_checkpoint: Path,
    checkpoint: Path,
    best_checkpoint: Path,
    training_result: ShogiPolicyValueTrainingResult | None,
    phase_timings: dict[str, float | None],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "schema_version": "intrep.shogi_online_replay_metrics.v1",
        "iteration_index": iteration_index,
        "iteration": {
            "wall_time_sec": phase_timings.get("iteration_wall_time_sec"),
        },
        "checkpoint": {
            "init_path": str(init_checkpoint),
            "init_id": _checkpoint_actor_id(init_checkpoint),
            "path": str(checkpoint),
            "id": _checkpoint_actor_id(checkpoint),
            "best_path": str(best_checkpoint),
            "best_id": _checkpoint_actor_id(best_checkpoint),
            "save_wall_time_sec": phase_timings.get("checkpoint_save_wall_time_sec"),
        },
        "replay": {
            "size": replay_size,
            "capacity": config.replay_capacity,
            "min_size": config.min_replay_size,
            "sampled_examples": sampled_examples,
            "sampled_examples_per_iteration": config.training_budget.sampled_examples_per_iteration,
            "max_seed_examples_per_iteration": config.training_budget.max_seed_examples_per_iteration,
            "seed_data_selection": (
                str(config.replay_seed_data_selection) if config.replay_seed_data_selection is not None else None
            ),
            "seed_eligible_examples": replay_seed_eligible_examples,
            "seed_loaded_examples": replay_seed_eligible_examples,
            "seed_sampled_examples": seed_sampled_examples,
            "generated_replay_size": generated_replay_size,
            "generated_sampled_examples": generated_sampled_examples,
            "preloaded_examples": 0,
            "sampling_wall_time_sec": phase_timings.get("replay_sampling_wall_time_sec"),
            "generated_tensorize_wall_time_sec": phase_timings.get("generated_tensorize_wall_time_sec"),
        },
        "generation": {
            "appended_examples": appended_examples,
            "generated_holdout_examples": 0,
            "wall_time_sec": phase_timings.get("generation_wall_time_sec"),
            "train_extraction_wall_time_sec": phase_timings.get("generated_train_extraction_wall_time_sec"),
            "summary_path": str(artifacts.generation_summary_path),
            "summary": _load_json_if_exists(artifacts.generation_summary_path),
        },
        "gate": {
            "wall_time_sec": phase_timings.get("gate_wall_time_sec"),
            "config": {
                "games": config.generator_gate_games,
                "worker_processes": config.generator_gate_worker_processes,
                "mcts_simulations": config.simulations,
                "nn_leaf_eval_batch_limit": config.nn_leaf_eval_batch_limit,
                "max_plies": config.max_plies,
            },
            "result_path": str(artifacts.generator_gate_result_path),
            "result": _load_json_if_exists(artifacts.generator_gate_result_path),
        },
        "training": {
            "skipped": training_skipped,
            "eval_data_selection": str(config.training_eval_data_selection),
            "eval_examples": training_eval_examples,
            "eval_source": "fixed_data_selection",
            "batch_size": config.training_budget.batch_size,
            "target_sample_passes": config.training_budget.target_sample_passes,
            "max_optimizer_steps_per_iteration": config.training_budget.max_optimizer_steps,
            "optimizer_steps_per_iteration": (
                training_result.metrics.actual_steps if training_result is not None else None
            ),
            "effective_sample_passes": _effective_sample_passes(training_result, config, sampled_examples),
            "wall_time_sec": phase_timings.get("training_wall_time_sec"),
            "config": asdict(training_result.config) if training_result is not None else None,
            "metrics": asdict(training_result.metrics) if training_result is not None else None,
        },
    }
    if skip_reason is not None:
        training = metrics["training"]
        assert isinstance(training, dict)
        training["skip_reason"] = skip_reason
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
    if config.iterations <= 0:
        raise ValueError("iterations must be positive")
    if config.next_checkpoint not in {"best", "final"}:
        raise ValueError("next_checkpoint must be best or final")
    if config.generator_gate_games <= 0:
        raise ValueError("generator_gate_games must be positive")
    if config.generator_gate_worker_processes <= 0:
        raise ValueError("generator_gate_worker_processes must be positive")
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
    if config.nn_leaf_eval_batch_limit <= 0:
        raise ValueError("nn_leaf_eval_batch_limit must be positive")
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
    if training_budget.sampled_examples_per_iteration <= 0:
        raise ValueError("sampled_examples_per_iteration must be positive")
    if training_budget.max_seed_examples_per_iteration < 0:
        raise ValueError("max_seed_examples_per_iteration must be non-negative")
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
    if source.policy_target_construction not in {
        "chosen_move",
        "decision_usi_multipv",
        "engine_analysis_multipv",
        "mcts_visit_counts",
    }:
        raise ValueError(
            "experience source policy_target_construction must be chosen_move, "
            "decision_usi_multipv, engine_analysis_multipv, or mcts_visit_counts"
        )
    if source.value_target_construction not in {"winner", "decision_usi_score", "engine_analysis_score"}:
        raise ValueError(
            "experience source value_target_construction must be winner, decision_usi_score, or engine_analysis_score"
        )
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


def _source_seed(base_seed: int, iteration_dir_name: str, source_index: int) -> int:
    iteration_index = int(iteration_dir_name.removeprefix("iteration-"))
    return base_seed + (iteration_index - 1) * 10000 + source_index


def _merge_jsonl(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as merged:
        for path in inputs:
            merged.write(path.read_text(encoding="utf-8"))


def _checkpoint_actor_id(checkpoint: Path) -> str:
    return load_shogi_policy_value_checkpoint_identity(checkpoint).checkpoint_id


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
    max_plies_draw_count = end_reasons.get("max_plies", 0)
    game_over_count = end_reasons.get("game_over", 0)
    return {
        "game_count": game_count,
        "end_reasons": end_reasons,
        "average_plies": total_plies / game_count if game_count else 0.0,
        "black_wins": black_wins,
        "white_wins": white_wins,
        "draws": draws,
        "max_plies_draw_count": max_plies_draw_count,
        "max_plies_draw_rate": max_plies_draw_count / game_count if game_count else 0.0,
        "game_over_count": game_over_count,
        "game_over_rate": game_over_count / game_count if game_count else 0.0,
        "generation_wall_time_sec": wall_time,
        "plies_per_sec": total_plies / wall_time if wall_time > 0.0 else 0.0,
        "sources": source_summaries,
    }


@dataclass(frozen=True)
class _ReplaySeedSelection:
    path: Path
    data_selection: ShogiPolicyValueDataSelection


def _load_replay_seed_selection(data_selection_path: Path | None) -> _ReplaySeedSelection | None:
    if data_selection_path is None:
        return None
    path = Path(data_selection_path)
    return _ReplaySeedSelection(path=path, data_selection=load_shogi_policy_value_data_selection(path))


def _count_replay_seed_examples(selection: _ReplaySeedSelection | None) -> int:
    if selection is None:
        return 0
    cache_path = default_shogi_policy_value_tensor_cache_path(selection.path)
    if cache_path.exists():
        return len(
            load_shogi_policy_value_tensor_cache(
                cache_path,
                expected_data_selection=selection.data_selection,
                expected_data_selection_root=selection.path.parent,
            ).train_samples
        )
    train_examples, _eval_examples = load_shogi_policy_value_data_selection_examples(selection.data_selection)
    return len(train_examples)


def _sample_replay_seed_samples(
    selection: _ReplaySeedSelection | None,
    *,
    sample_count: int,
    seed: int,
) -> list[LegalMovePolicyValueTensorSample]:
    if sample_count <= 0 or selection is None:
        return []
    cache_path = default_shogi_policy_value_tensor_cache_path(selection.path)
    if cache_path.exists():
        cache = load_shogi_policy_value_tensor_cache(
            cache_path,
            expected_data_selection=selection.data_selection,
            expected_data_selection_root=selection.path.parent,
        )
        return _sample_sequence(cache.train_samples, sample_count=sample_count, seed=seed)
    return tensorize_legal_move_policy_value_examples(
        _sample_replay_seed_examples_from_selection(selection.data_selection, sample_count=sample_count, seed=seed)
    )


def _sample_sequence(
    samples,
    *,
    sample_count: int,
    seed: int,
) -> list[LegalMovePolicyValueTensorSample]:
    if hasattr(samples, "shards") and hasattr(samples, "offsets"):
        return _sample_sharded_sequence(samples, sample_count=sample_count, seed=seed)
    sample_count = min(sample_count, len(samples))
    generator = torch.Generator().manual_seed(seed)
    indices = sorted(torch.randperm(len(samples), generator=generator)[:sample_count].tolist())
    return [samples[index] for index in indices]


def _sample_sharded_sequence(
    samples,
    *,
    sample_count: int,
    seed: int,
) -> list[LegalMovePolicyValueTensorSample]:
    sample_count = min(sample_count, len(samples))
    if sample_count <= 0:
        return []
    shard_counts = [int(shard["sample_count"]) for shard in samples.shards]
    if not shard_counts:
        return []
    average_shard_count = max(1.0, sum(shard_counts) / len(shard_counts))
    selected_shard_count = min(len(shard_counts), max(1, math.ceil(sample_count / average_shard_count) + 1))
    generator = torch.Generator().manual_seed(seed)
    weights = torch.tensor(shard_counts, dtype=torch.float64)
    shard_order = torch.multinomial(weights, len(shard_counts), replacement=False, generator=generator).tolist()
    selected_shards = sorted(shard_order[:selected_shard_count])
    selected_ranges = [
        (int(samples.offsets[shard_index]), int(samples.offsets[shard_index]) + shard_counts[shard_index])
        for shard_index in selected_shards
    ]
    selected_pool_size = sum(end - start for start, end in selected_ranges)
    sample_count = min(sample_count, selected_pool_size)
    local_indices = sorted(torch.randperm(selected_pool_size, generator=generator)[:sample_count].tolist())
    global_indices: list[int] = []
    range_index = 0
    range_start = 0
    for local_index in local_indices:
        while range_index + 1 < len(selected_ranges) and local_index >= range_start + (
            selected_ranges[range_index][1] - selected_ranges[range_index][0]
        ):
            range_start += selected_ranges[range_index][1] - selected_ranges[range_index][0]
            range_index += 1
        start, _end = selected_ranges[range_index]
        global_indices.append(start + local_index - range_start)
    return [samples[index] for index in global_indices]


def _sample_replay_seed_examples_from_selection(
    selection: ShogiPolicyValueDataSelection | _ReplaySeedSelection | None,
    *,
    sample_count: int,
    seed: int,
) -> list[ShogiMovePolicyValueExample]:
    if isinstance(selection, _ReplaySeedSelection):
        selection = selection.data_selection
    if sample_count <= 0 or selection is None:
        return []
    examples_by_source = []
    for source in selection.train_sources:
        if source.kind == "shogi_policy_value_examples_jsonl":
            examples_by_source.append(load_shogi_move_policy_value_examples_jsonl(source.path, max_examples=source.max_examples))
        else:
            source_selection = ShogiPolicyValueDataSelection(
                name=selection.name,
                objective=selection.objective,
                target_construction=selection.target_construction,
                analysis_sources=selection.analysis_sources,
                train_sources=(source,),
                eval_sources=selection.eval_sources,
            )
            source_examples, _eval_examples = load_shogi_policy_value_data_selection_examples(source_selection)
            examples_by_source.append(source_examples)
    total_examples = sum(len(examples) for examples in examples_by_source)
    if total_examples == 0:
        return []
    sample_count = min(sample_count, total_examples)
    generator = torch.Generator().manual_seed(seed)
    selected_positions = sorted(torch.randperm(total_examples, generator=generator)[:sample_count].tolist())
    sampled: list[ShogiMovePolicyValueExample] = []
    source_index = 0
    source_start = 0
    for position in selected_positions:
        while source_index + 1 < len(examples_by_source) and position >= source_start + len(examples_by_source[source_index]):
            source_start += len(examples_by_source[source_index])
            source_index += 1
        sampled.append(examples_by_source[source_index][position - source_start])
    return sampled


def _load_training_eval_samples(data_selection_path: Path) -> Sequence[LegalMovePolicyValueTensorSample]:
    selection_path = Path(data_selection_path)
    selection = load_shogi_policy_value_data_selection(selection_path)
    cache_path = default_shogi_policy_value_tensor_cache_path(selection_path)
    if cache_path.exists():
        return load_shogi_policy_value_tensor_cache(
            cache_path,
            expected_data_selection=selection,
            expected_data_selection_root=selection_path.parent,
        ).eval_samples
    _train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(selection)
    return tensorize_legal_move_policy_value_examples(eval_examples)


def _load_generated_policy_value_examples(
    path: Path,
    *,
    policy_target_construction: str,
    value_target_construction: str,
) -> list[ShogiMovePolicyValueExample]:
    return load_shogi_move_policy_value_examples_from_game_records_jsonl(
        path,
        policy_target_construction=policy_target_construction,
        value_target_construction=value_target_construction,
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
        assembly_spec_id=checkpoint_config.assembly_spec_id,
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
