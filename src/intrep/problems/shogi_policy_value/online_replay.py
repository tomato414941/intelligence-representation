from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

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
    run_shogi_generated_games,
    warn_short_max_plies,
)
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingResult,
    train_shogi_policy_value_model,
)
from intrep.worlds.shogi.experience_store import append_shogi_experience_store
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl

DEFAULT_REPLAY_CAPACITY = 32768
DEFAULT_REPLAY_SAMPLE_SIZE = 4096
DEFAULT_MIN_REPLAY_SIZE = 8192


@dataclass(frozen=True)
class ShogiGeneratedExperienceSource:
    name: str
    opponent: str
    games: int
    usi_command: str | None = None
    usi_options: tuple[str, ...] = ()
    usi_go_command: str = "go nodes 1"


@dataclass(frozen=True)
class ShogiOnlineReplayConfig:
    checkpoint: Path
    run_dir: Path
    cycles: int = 1
    replay_capacity: int = DEFAULT_REPLAY_CAPACITY
    replay_sample_size: int = DEFAULT_REPLAY_SAMPLE_SIZE
    min_replay_size: int = DEFAULT_MIN_REPLAY_SIZE
    experience_store_dir: Path | None = None
    replay_seed_data_selection: Path | None = None
    training_eval_data_selection: Path | None = None
    next_checkpoint: str = "best"
    arena_repo: Path = Path("../shogi-arena-agent")
    experience_sources: tuple[ShogiGeneratedExperienceSource, ...] = (
        ShogiGeneratedExperienceSource(name="self-play", opponent="self", games=4),
    )
    concurrent_games_per_process: int = 1
    generation_progress_every_plies: int = 0
    board_backend: str = "cshogi"
    max_plies: int = DEFAULT_SHOGI_MAX_PLIES
    simulations: int = 16
    evaluation_batch_size: int = 1
    generation_worker_processes: int = 1
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0
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
    training_eval_data_selection: Path | None
    preloaded_examples: int
    fixed_eval_examples: int
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
            "training_eval_data_selection": str(self.training_eval_data_selection) if self.training_eval_data_selection is not None else None,
            "preloaded_examples": self.preloaded_examples,
            "fixed_eval_examples": self.fixed_eval_examples,
            "cycles": [cycle.to_json() for cycle in self.cycles],
        }


@dataclass(frozen=True)
class ShogiOnlineReplayCycleArtifacts:
    cycle_dir: Path
    games_jsonl: Path
    train_jsonl: Path
    eval_jsonl: Path
    generation_summary_path: Path
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
    fixed_eval_examples = _load_training_eval_examples(config.training_eval_data_selection)
    fixed_eval_samples = tensorize_shogi_policy_value_examples(fixed_eval_examples)
    replay.extend(tensorize_shogi_policy_value_examples(preloaded_examples))
    generator = torch.Generator().manual_seed(config.seed)
    cycle_results: list[ShogiOnlineReplayCycleResult] = []
    for cycle_index in range(1, config.cycles + 1):
        artifacts = _online_replay_cycle_artifacts(run_dir, cycle_index)
        _generate_online_replay_cycle_experience(config=config, checkpoint=checkpoint, artifacts=artifacts)
        experience_store_append = _append_to_experience_store(
            store_dir=config.experience_store_dir,
            games_jsonl=artifacts.games_jsonl,
        )
        new_examples, generated_eval_examples = _load_online_replay_cycle_examples(
            artifacts=artifacts,
            eval_ratio=config.eval_ratio,
        )
        eval_samples = fixed_eval_samples or tensorize_shogi_policy_value_examples(generated_eval_examples)
        replay.extend(tensorize_shogi_policy_value_examples(new_examples))
        if len(replay) < config.min_replay_size:
            sampled_examples: list[TensorizedShogiPolicyValueSample] = []
            training_skipped = True
            effective_checkpoint = checkpoint
            effective_best_checkpoint = checkpoint
            training_result = None
            skip_reason = "min_replay_size"
        else:
            sampled_examples = replay.sample(min(config.replay_sample_size, len(replay)), generator=generator)
            training_result = _train_online_replay_cycle(
                config=config,
                checkpoint=checkpoint,
                artifacts=artifacts,
                sampled_examples=sampled_examples,
                eval_examples=eval_samples,
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
            fixed_eval_examples=len(fixed_eval_examples),
            generated_eval_examples=len(generated_eval_examples),
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
        fixed_eval_examples=len(fixed_eval_examples),
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
        train_jsonl=cycle_dir / "train-games.jsonl",
        eval_jsonl=cycle_dir / "eval-games.jsonl",
        generation_summary_path=cycle_dir / "generation-summary.json",
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
            opponent=source.opponent,
            usi_command=source.usi_command,
            usi_options=source.usi_options,
            usi_go_command=source.usi_go_command,
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
            checkpoint_device=config.device,
            mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
        )
        source_game_paths.append(games_jsonl)
        source_summaries.append(
            {
                "name": source.name,
                "opponent": source.opponent,
                "games": source.games,
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


def _load_online_replay_cycle_examples(
    *,
    artifacts: ShogiOnlineReplayCycleArtifacts,
    eval_ratio: float,
) -> tuple[list[ShogiPolicyValueExample], list[ShogiPolicyValueExample]]:
    split_shogi_game_records_jsonl(
        games_jsonl=artifacts.games_jsonl,
        train_jsonl=artifacts.train_jsonl,
        eval_jsonl=artifacts.eval_jsonl,
        eval_ratio=eval_ratio,
    )
    new_examples = _load_generated_policy_value_examples(artifacts.train_jsonl)
    generated_eval_examples = _load_generated_policy_value_examples(artifacts.eval_jsonl)
    return new_examples, generated_eval_examples


def _train_online_replay_cycle(
    *,
    config: ShogiOnlineReplayConfig,
    checkpoint: Path,
    artifacts: ShogiOnlineReplayCycleArtifacts,
    sampled_examples: list[TensorizedShogiPolicyValueSample],
    eval_examples: list[TensorizedShogiPolicyValueSample],
) -> ShogiPolicyValueTrainingResult:
    training_config = _training_config_from_checkpoint(checkpoint, config)
    training_result = train_shogi_policy_value_model(
        sampled_examples,
        eval_examples=eval_examples,
        config=training_config,
        initial_state_dict=load_shogi_policy_value_checkpoint_state_dict(checkpoint, device=config.device),
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
    fixed_eval_examples: int,
    generated_eval_examples: int,
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
        "training_eval_data_selection": str(config.training_eval_data_selection) if config.training_eval_data_selection is not None else None,
        "preloaded_examples": preloaded_examples,
        "fixed_eval_examples": fixed_eval_examples,
        "generated_eval_examples": generated_eval_examples,
        "training_eval_source": "fixed" if fixed_eval_examples else "generated_cycle",
        "experience_store_append": experience_store_append,
        "generation_summary_path": str(artifacts.generation_summary_path),
        "generation_summary": _load_json_if_exists(artifacts.generation_summary_path),
        "sampled_examples": sampled_examples,
        "init_checkpoint_path": str(init_checkpoint),
        "checkpoint_path": str(checkpoint),
        "best_checkpoint_path": str(best_checkpoint),
        "config": asdict(training_result.config) if training_result is not None else None,
        "metrics": asdict(training_result.metrics) if training_result is not None else None,
    }
    if skip_reason is not None:
        metrics["skip_reason"] = skip_reason
    return metrics


def _load_json_if_exists(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_online_replay_config(config: ShogiOnlineReplayConfig) -> None:
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.replay_sample_size <= 0:
        raise ValueError("replay_sample_size must be positive")
    if config.min_replay_size <= 0:
        raise ValueError("min_replay_size must be positive")
    if config.min_replay_size > config.replay_capacity:
        raise ValueError("min_replay_size must be less than or equal to replay_capacity")
    if config.cycles <= 0:
        raise ValueError("cycles must be positive")
    if config.next_checkpoint not in {"best", "final"}:
        raise ValueError("next_checkpoint must be best or final")
    if not config.experience_sources:
        raise ValueError("experience_sources must not be empty")
    for source in config.experience_sources:
        _validate_experience_source(source)
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
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")


def _validate_experience_source(source: ShogiGeneratedExperienceSource) -> None:
    if not source.name:
        raise ValueError("experience source name must not be empty")
    if not all(character.isalnum() or character in "-_" for character in source.name):
        raise ValueError("experience source name must contain only letters, numbers, hyphen, or underscore")
    if source.opponent not in {"self", "usi"}:
        raise ValueError("experience source opponent must be self or usi")
    if source.opponent == "usi" and not source.usi_command:
        raise ValueError("usi_command is required when experience source opponent is usi")
    if source.games <= 0:
        raise ValueError("experience source games must be positive")


def _source_seed(base_seed: int, cycle_dir_name: str, source_index: int) -> int:
    cycle_index = int(cycle_dir_name.removeprefix("cycle-"))
    return base_seed + (cycle_index - 1) * 10000 + source_index


def _merge_jsonl(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as merged:
        for path in inputs:
            merged.write(path.read_text(encoding="utf-8"))


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


def _load_training_eval_examples(data_selection_path: Path | None) -> list[ShogiPolicyValueExample]:
    if data_selection_path is None:
        return []
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
) -> ShogiPolicyValueTrainingConfig:
    checkpoint_config = load_shogi_policy_value_checkpoint_training_config(checkpoint, device=config.device)
    return ShogiPolicyValueTrainingConfig(
        max_steps=config.max_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        embedding_dim=checkpoint_config.embedding_dim,
        hidden_dim=checkpoint_config.hidden_dim,
        num_heads=checkpoint_config.num_heads,
        num_layers=checkpoint_config.num_layers,
        use_shared_core=checkpoint_config.use_shared_core,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        device=config.device,
        num_workers=config.num_workers,
    )
