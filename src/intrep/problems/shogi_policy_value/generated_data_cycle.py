from __future__ import annotations

import json
import subprocess
import sys
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
    ShogiPolicyValueDataSelection,
    ShogiPolicyValueDataSelectionSource,
    ShogiPolicyValueTargetConstruction,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.examples import ShogiPolicyValueExample
from intrep.problems.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    train_shogi_policy_value_model,
)
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingCycleConfig:
    checkpoint: Path
    run_dir: Path
    arena_repo: Path = Path("../shogi-arena-agent")
    opponent: str = "self"
    yaneuraou: str | None = None
    engine_go_command: str = "go nodes 1"
    games: int = 4
    max_plies: int = 80
    simulations: int = 16
    evaluation_batch_size: int = 1
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingLoopConfig:
    checkpoint: Path
    run_dir: Path
    cycles: int = 1
    next_checkpoint: str = "best"
    arena_repo: Path = Path("../shogi-arena-agent")
    opponent: str = "self"
    yaneuraou: str | None = None
    engine_go_command: str = "go nodes 1"
    games: int = 4
    max_plies: int = 80
    simulations: int = 16
    evaluation_batch_size: int = 1
    mcts_move_time_limit_sec: float | None = None
    eval_ratio: float = 0.25
    max_steps: int = 100
    batch_size: int = 128
    learning_rate: float = 0.0005
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0


@dataclass(frozen=True)
class ShogiOnlineReplayConfig:
    checkpoint: Path
    run_dir: Path
    cycles: int = 1
    replay_capacity: int = 1024
    replay_sample_size: int = 128
    next_checkpoint: str = "best"
    arena_repo: Path = Path("../shogi-arena-agent")
    opponent: str = "self"
    yaneuraou: str | None = None
    engine_go_command: str = "go nodes 1"
    games: int = 4
    max_plies: int = 80
    simulations: int = 16
    evaluation_batch_size: int = 1
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
class ShogiGeneratedDataTrainingCycleResult:
    run_dir: Path
    generated_games_jsonl: Path
    train_games: int
    eval_games: int
    data_selection: Path
    checkpoint: Path
    best_checkpoint: Path
    metrics: Path
    generation: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "generated_games_jsonl": str(self.generated_games_jsonl),
            "train_games": self.train_games,
            "eval_games": self.eval_games,
            "data_selection": str(self.data_selection),
            "checkpoint": str(self.checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "metrics": str(self.metrics),
            "generation": self.generation,
        }


@dataclass(frozen=True)
class ShogiGeneratedDataTrainingLoopResult:
    run_dir: Path
    initial_checkpoint: Path
    final_checkpoint: Path
    next_checkpoint: str
    cycles: tuple[ShogiGeneratedDataTrainingCycleResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "final_checkpoint": str(self.final_checkpoint),
            "next_checkpoint": self.next_checkpoint,
            "cycles": [cycle.to_json() for cycle in self.cycles],
        }


@dataclass(frozen=True)
class ShogiOnlineReplayCycleResult:
    cycle_index: int
    run_dir: Path
    generated_games_jsonl: Path
    appended_examples: int
    replay_size: int
    sampled_examples: int
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
    cycles: tuple[ShogiOnlineReplayCycleResult, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "run_dir": str(self.run_dir),
            "initial_checkpoint": str(self.initial_checkpoint),
            "final_checkpoint": str(self.final_checkpoint),
            "next_checkpoint": self.next_checkpoint,
            "replay_capacity": self.replay_capacity,
            "cycles": [cycle.to_json() for cycle in self.cycles],
        }


def run_shogi_generated_data_training_cycle(
    config: ShogiGeneratedDataTrainingCycleConfig,
) -> ShogiGeneratedDataTrainingCycleResult:
    _validate_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    games_jsonl = run_dir / "generated-games.jsonl"
    train_jsonl = run_dir / "train-games.jsonl"
    eval_jsonl = run_dir / "eval-games.jsonl"
    data_selection_json = run_dir / "data-selection.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best-checkpoint.pt"
    metrics_path = run_dir / "metrics.json"

    _run_generate_games(
        arena_repo=config.arena_repo,
        checkpoint=config.checkpoint,
        opponent=config.opponent,
        yaneuraou=config.yaneuraou,
        engine_go_command=config.engine_go_command,
        out=games_jsonl,
        games=config.games,
        max_plies=config.max_plies,
        simulations=config.simulations,
        evaluation_batch_size=config.evaluation_batch_size,
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
        device=config.device,
        num_workers=config.num_workers,
    )
    return ShogiGeneratedDataTrainingCycleResult(
        run_dir=run_dir,
        generated_games_jsonl=games_jsonl,
        train_games=train_count,
        eval_games=eval_count,
        data_selection=data_selection_json,
        checkpoint=checkpoint_path,
        best_checkpoint=best_checkpoint_path,
        metrics=metrics_path,
        generation={
            "opponent": config.opponent,
            "games": config.games,
            "max_plies": config.max_plies,
            "simulations": config.simulations,
            "evaluation_batch_size": config.evaluation_batch_size,
            "mcts_move_time_limit_sec": config.mcts_move_time_limit_sec,
        },
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
                opponent=config.opponent,
                yaneuraou=config.yaneuraou,
                engine_go_command=config.engine_go_command,
                games=config.games,
                max_plies=config.max_plies,
                simulations=config.simulations,
                evaluation_batch_size=config.evaluation_batch_size,
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
        results.append(result)
        checkpoint = _promoted_checkpoint(result, policy=config.next_checkpoint)
    return ShogiGeneratedDataTrainingLoopResult(
        run_dir=run_dir,
        initial_checkpoint=config.checkpoint,
        final_checkpoint=checkpoint,
        next_checkpoint=config.next_checkpoint,
        cycles=tuple(results),
    )


def run_shogi_online_replay(
    config: ShogiOnlineReplayConfig,
) -> ShogiOnlineReplayResult:
    _validate_online_replay_config(config)
    run_dir = config.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = config.checkpoint
    replay = ReplayBuffer[ShogiPolicyValueExample](capacity=config.replay_capacity)
    generator = torch.Generator().manual_seed(config.seed)
    cycle_results: list[ShogiOnlineReplayCycleResult] = []
    for cycle_index in range(1, config.cycles + 1):
        cycle_dir = run_dir / f"cycle-{cycle_index:04d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        games_jsonl = cycle_dir / "generated-games.jsonl"
        train_jsonl = cycle_dir / "train-games.jsonl"
        eval_jsonl = cycle_dir / "eval-games.jsonl"
        checkpoint_path = cycle_dir / "checkpoint.pt"
        best_checkpoint_path = cycle_dir / "best-checkpoint.pt"
        metrics_path = cycle_dir / "metrics.json"
        _run_generate_games(
            arena_repo=config.arena_repo,
            checkpoint=checkpoint,
            opponent=config.opponent,
            yaneuraou=config.yaneuraou,
            engine_go_command=config.engine_go_command,
            out=games_jsonl,
            games=config.games,
            max_plies=config.max_plies,
            simulations=config.simulations,
            evaluation_batch_size=config.evaluation_batch_size,
            mcts_move_time_limit_sec=config.mcts_move_time_limit_sec,
        )
        split_shogi_game_records_jsonl(
            games_jsonl=games_jsonl,
            train_jsonl=train_jsonl,
            eval_jsonl=eval_jsonl,
            eval_ratio=config.eval_ratio,
        )
        new_examples = _load_generated_policy_value_examples(train_jsonl)
        eval_examples = _load_generated_policy_value_examples(eval_jsonl)
        replay.extend(new_examples)
        sampled_examples = replay.sample(min(config.replay_sample_size, len(replay)), generator=generator)
        training_config = _training_config_from_checkpoint(checkpoint, config)
        training_result = train_shogi_policy_value_model(
            sampled_examples,
            eval_examples=eval_examples,
            config=training_config,
            initial_state_dict=load_shogi_policy_value_checkpoint_state_dict(checkpoint, device=config.device),
        )
        save_shogi_policy_value_checkpoint(checkpoint_path, training_result)
        if training_result.best_model_state_dict is not None:
            save_shogi_policy_value_state_checkpoint(
                best_checkpoint_path,
                training_result.best_model_state_dict,
                training_result.config,
            )
        else:
            save_shogi_policy_value_checkpoint(best_checkpoint_path, training_result)
        metrics = {
            "schema": "shogi_online_replay_v1",
            "cycle_index": cycle_index,
            "appended_examples": len(new_examples),
            "replay_size": len(replay),
            "sampled_examples": len(sampled_examples),
            "init_checkpoint_path": str(checkpoint),
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "config": asdict(training_result.config),
            "metrics": asdict(training_result.metrics),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cycle_result = ShogiOnlineReplayCycleResult(
            cycle_index=cycle_index,
            run_dir=cycle_dir,
            generated_games_jsonl=games_jsonl,
            appended_examples=len(new_examples),
            replay_size=len(replay),
            sampled_examples=len(sampled_examples),
            checkpoint=checkpoint_path,
            best_checkpoint=best_checkpoint_path,
            metrics=metrics_path,
        )
        cycle_results.append(cycle_result)
        checkpoint = _promoted_online_replay_checkpoint(cycle_result, policy=config.next_checkpoint)
    return ShogiOnlineReplayResult(
        run_dir=run_dir,
        initial_checkpoint=config.checkpoint,
        final_checkpoint=checkpoint,
        next_checkpoint=config.next_checkpoint,
        replay_capacity=config.replay_capacity,
        cycles=tuple(cycle_results),
    )


def _validate_config(config: ShogiGeneratedDataTrainingCycleConfig) -> None:
    if config.opponent not in {"self", "yaneuraou"}:
        raise ValueError("opponent must be self or yaneuraou")
    if config.opponent == "yaneuraou" and not config.yaneuraou:
        raise ValueError("yaneuraou is required when opponent is yaneuraou")
    if config.games <= 0:
        raise ValueError("games must be positive")
    if config.max_plies <= 0:
        raise ValueError("max_plies must be positive")
    if config.simulations <= 0:
        raise ValueError("simulations must be positive")
    if config.evaluation_batch_size <= 0:
        raise ValueError("evaluation_batch_size must be positive")
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
            opponent=config.opponent,
            yaneuraou=config.yaneuraou,
            engine_go_command=config.engine_go_command,
            games=config.games,
            max_plies=config.max_plies,
            simulations=config.simulations,
            evaluation_batch_size=config.evaluation_batch_size,
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


def _validate_online_replay_config(config: ShogiOnlineReplayConfig) -> None:
    if config.replay_capacity <= 0:
        raise ValueError("replay_capacity must be positive")
    if config.replay_sample_size <= 0:
        raise ValueError("replay_sample_size must be positive")
    _validate_loop_config(
        ShogiGeneratedDataTrainingLoopConfig(
            checkpoint=config.checkpoint,
            run_dir=config.run_dir,
            cycles=config.cycles,
            next_checkpoint=config.next_checkpoint,
            arena_repo=config.arena_repo,
            opponent=config.opponent,
            yaneuraou=config.yaneuraou,
            engine_go_command=config.engine_go_command,
            games=config.games,
            max_plies=config.max_plies,
            simulations=config.simulations,
            evaluation_batch_size=config.evaluation_batch_size,
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


def _promoted_checkpoint(result: ShogiGeneratedDataTrainingCycleResult, *, policy: str) -> Path:
    if policy == "best":
        return result.best_checkpoint
    if policy == "final":
        return result.checkpoint
    raise ValueError("next_checkpoint must be best or final")


def _promoted_online_replay_checkpoint(result: ShogiOnlineReplayCycleResult, *, policy: str) -> Path:
    if policy == "best":
        return result.best_checkpoint
    if policy == "final":
        return result.checkpoint
    raise ValueError("next_checkpoint must be best or final")


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


def _run_generate_games(
    *,
    arena_repo: Path,
    checkpoint: Path,
    opponent: str,
    yaneuraou: str | None,
    engine_go_command: str,
    out: Path,
    games: int,
    max_plies: int,
    simulations: int,
    evaluation_batch_size: int,
    mcts_move_time_limit_sec: float | None,
) -> None:
    if opponent == "yaneuraou" and not yaneuraou:
        raise SystemExit("--yaneuraou is required when --opponent yaneuraou")

    command = [
        "uv",
        "run",
        "python",
        "scripts/generate_shogi_games.py",
        "--black-kind",
        "checkpoint",
        "--black-checkpoint",
        str(checkpoint.resolve()),
        "--black-checkpoint-policy",
        "mcts",
        "--black-checkpoint-simulations",
        str(simulations),
        "--black-checkpoint-evaluation-batch-size",
        str(evaluation_batch_size),
        "--games",
        str(games),
        "--max-plies",
        str(max_plies),
        "--out",
        str(out),
    ]
    if mcts_move_time_limit_sec is not None:
        command.extend(["--black-checkpoint-move-time-limit-sec", str(mcts_move_time_limit_sec)])
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
                "--white-checkpoint-policy",
                "mcts",
                "--white-checkpoint-simulations",
                str(simulations),
                "--white-checkpoint-evaluation-batch-size",
                str(evaluation_batch_size),
            ]
        )
        if mcts_move_time_limit_sec is not None:
            command.extend(["--white-checkpoint-move-time-limit-sec", str(mcts_move_time_limit_sec)])
    subprocess.run(command, cwd=arena_repo.resolve(), check=True)


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
    subprocess.run(command, check=True)
