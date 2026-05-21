from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from intrep.problems.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint_training_config,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    DEFAULT_SHOGI_MAX_PLIES,
    DEFAULT_USI_READ_TIMEOUT_SECONDS,
    checkpoint_generated_player,
    usi_engine_generated_player,
)
from intrep.problems.shogi_policy_value.online_replay import (
    DEFAULT_GENERATOR_GATE_GAMES,
    DEFAULT_GENERATOR_GATE_WORKER_PROCESSES,
    DEFAULT_GENERATION_WORKER_PROCESSES,
    DEFAULT_MAX_SEED_EXAMPLES_PER_ITERATION,
    DEFAULT_MIN_REPLAY_SIZE,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION,
    DEFAULT_TARGET_SAMPLE_PASSES,
    DEFAULT_TRAINING_BATCH_SIZE,
    ShogiOnlineReplayConfig,
    ShogiGeneratedExperienceSource,
    ShogiOnlineReplayTrainingBudget,
    run_shogi_online_replay,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run shogi Online Experience Replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--min-replay-size", type=int, default=DEFAULT_MIN_REPLAY_SIZE)
    parser.add_argument("--sampled-examples-per-iteration", type=int, default=DEFAULT_SAMPLED_EXAMPLES_PER_ITERATION)
    parser.add_argument("--max-seed-examples-per-iteration", type=int, default=DEFAULT_MAX_SEED_EXAMPLES_PER_ITERATION)
    parser.add_argument("--training-batch-size", type=int, default=DEFAULT_TRAINING_BATCH_SIZE)
    parser.add_argument("--target-sample-passes", type=float, default=DEFAULT_TARGET_SAMPLE_PASSES)
    parser.add_argument("--max-optimizer-steps-per-iteration", type=int)
    parser.add_argument("--generator-gate-games", type=int, default=DEFAULT_GENERATOR_GATE_GAMES)
    parser.add_argument("--generator-gate-worker-processes", type=int, default=DEFAULT_GENERATOR_GATE_WORKER_PROCESSES)
    parser.add_argument("--replay-seed-data-selection", type=Path)
    parser.add_argument("--training-eval-data-selection", type=Path, required=True)
    parser.add_argument("--next-checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument(
        "--experience-source",
        action="append",
        required=True,
        metavar="KIND:GAMES",
        help=(
            "Generated experience source. Repeatable. KIND is checkpoint-self, "
            "checkpoint-black-vs-usi, usi-black-vs-checkpoint, or checkpoint-vs-usi-balanced."
        ),
    )
    parser.add_argument("--usi-command", help="USI engine command used by usi_engine players.")
    parser.add_argument("--usi-option", action="append", default=[], help="USI engine option as NAME=VALUE.")
    parser.add_argument("--usi-go-command", default="go nodes 1")
    parser.add_argument("--usi-read-timeout-seconds", type=float, default=DEFAULT_USI_READ_TIMEOUT_SECONDS)
    parser.add_argument("--checkpoint-move-selection-profile", choices=("visit-sampling",), default="visit-sampling")
    parser.add_argument("--checkpoint-move-selection-temperature", type=float)
    parser.add_argument("--checkpoint-move-selection-temperature-plies", type=int)
    parser.add_argument("--concurrent-games-per-process", type=int, default=1)
    parser.add_argument("--generation-progress-every-plies", type=int, default=0)
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--nn-leaf-eval-batch-limit", type=int, default=64)
    parser.add_argument("--generation-worker-processes", type=int, default=DEFAULT_GENERATION_WORKER_PROCESSES)
    parser.add_argument("--mcts-move-time-limit-sec", type=float)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--allow-nonstandard-loss-weights", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-eval-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--progress-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    checkpoint_training_config = load_shogi_policy_value_checkpoint_training_config(
        args.checkpoint,
        device=args.device,
    )

    result = run_shogi_online_replay(
        ShogiOnlineReplayConfig(
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            iterations=args.iterations,
            resume=args.resume,
            replay_capacity=args.replay_capacity,
            min_replay_size=args.min_replay_size,
            training_budget=ShogiOnlineReplayTrainingBudget(
                sampled_examples_per_iteration=args.sampled_examples_per_iteration,
                max_seed_examples_per_iteration=args.max_seed_examples_per_iteration,
                batch_size=args.training_batch_size,
                target_sample_passes=args.target_sample_passes,
                max_optimizer_steps=args.max_optimizer_steps_per_iteration,
            ),
            generator_gate_games=args.generator_gate_games,
            generator_gate_worker_processes=args.generator_gate_worker_processes,
            replay_seed_data_selection=args.replay_seed_data_selection,
            training_eval_data_selection=args.training_eval_data_selection,
            next_checkpoint=args.next_checkpoint,
            arena_repo=args.arena_repo,
            experience_sources=_experience_sources_from_args(args),
            concurrent_games_per_process=args.concurrent_games_per_process,
            generation_progress_every_plies=args.generation_progress_every_plies,
            board_backend=args.board_backend,
            max_plies=args.max_plies,
            simulations=args.simulations,
            nn_leaf_eval_batch_limit=args.nn_leaf_eval_batch_limit,
            generation_worker_processes=args.generation_worker_processes,
            mcts_move_time_limit_sec=args.mcts_move_time_limit_sec,
            training_config=replace(
                checkpoint_training_config,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                policy_loss_weight=args.policy_loss_weight,
                value_loss_weight=args.value_loss_weight,
                allow_nonstandard_loss_weights=args.allow_nonstandard_loss_weights,
                device=args.device,
                max_train_eval_examples=args.max_train_eval_examples,
                max_eval_examples=args.max_eval_examples,
                log_every=args.log_every,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                progress_every=args.progress_every,
                eval_every=args.eval_every,
                early_stopping_patience=args.early_stopping_patience,
                seed=args.seed,
            ),
            seed=args.seed,
        )
    )
    print(json.dumps(result.to_json(), indent=2))


def _experience_sources_from_args(args: argparse.Namespace) -> tuple[ShogiGeneratedExperienceSource, ...]:
    sources = []
    for index, value in enumerate(args.experience_source):
        kind, separator, games = value.partition(":")
        if not separator:
            raise SystemExit("--experience-source must be KIND:GAMES")
        game_count = int(games)
        if kind == "checkpoint-self":
            sources.append(
                ShogiGeneratedExperienceSource(
                    name=f"checkpoint-self-{index}",
                    games=game_count,
                    black_player=_checkpoint_player(args, name="black"),
                    white_player=_checkpoint_player(args, name="white"),
                    policy_target_construction="mcts_visit_counts",
                    value_target_construction="winner",
                )
            )
            continue
        if kind == "checkpoint-black-vs-usi":
            sources.append(_checkpoint_vs_usi_source(args, name=f"checkpoint-black-vs-usi-{index}", games=game_count))
            continue
        if kind == "usi-black-vs-checkpoint":
            sources.append(_usi_vs_checkpoint_source(args, name=f"usi-black-vs-checkpoint-{index}", games=game_count))
            continue
        if kind == "checkpoint-vs-usi-balanced":
            black_games = game_count // 2 + game_count % 2
            white_games = game_count // 2
            sources.append(_checkpoint_vs_usi_source(args, name=f"checkpoint-black-vs-usi-{index}", games=black_games))
            if white_games:
                sources.append(_usi_vs_checkpoint_source(args, name=f"usi-black-vs-checkpoint-{index}", games=white_games))
            continue
        raise SystemExit(
            "--experience-source KIND must be checkpoint-self, checkpoint-black-vs-usi, "
            "usi-black-vs-checkpoint, or checkpoint-vs-usi-balanced"
        )
    return tuple(sources)


def _usi_player(args: argparse.Namespace, *, name: str):
    if not args.usi_command:
        raise SystemExit("--usi-command is required for usi_engine experience sources")
    return usi_engine_generated_player(
        name=name,
        command=args.usi_command,
        options=tuple(args.usi_option),
        go_command=args.usi_go_command,
        read_timeout_seconds=args.usi_read_timeout_seconds,
    )


def _checkpoint_player(args: argparse.Namespace, *, name: str):
    temperature = args.checkpoint_move_selection_temperature
    temperature_plies = args.checkpoint_move_selection_temperature_plies
    temperature = 1.0 if temperature is None else temperature
    temperature_plies = 40 if temperature_plies is None else temperature_plies
    return checkpoint_generated_player(
        name,
        move_selection_profile=args.checkpoint_move_selection_profile,
        move_selection_temperature=temperature,
        move_selection_temperature_plies=temperature_plies,
    )


def _checkpoint_vs_usi_source(args: argparse.Namespace, *, name: str, games: int) -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=_checkpoint_player(args, name="checkpoint"),
        white_player=_usi_player(args, name="usi_engine"),
        policy_target_construction="chosen_move",
        value_target_construction="winner",
    )


def _usi_vs_checkpoint_source(args: argparse.Namespace, *, name: str, games: int) -> ShogiGeneratedExperienceSource:
    return ShogiGeneratedExperienceSource(
        name=name,
        games=games,
        black_player=_usi_player(args, name="usi_engine"),
        white_player=_checkpoint_player(args, name="checkpoint"),
        policy_target_construction="chosen_move",
        value_target_construction="winner",
    )


if __name__ == "__main__":
    main()
