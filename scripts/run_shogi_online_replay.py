from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.generated_data_cycle import (
    DEFAULT_SHOGI_MAX_PLIES,
    DEFAULT_MIN_REPLAY_SIZE,
    DEFAULT_REPLAY_CAPACITY,
    DEFAULT_REPLAY_SAMPLE_SIZE,
    ShogiOnlineReplayConfig,
    ShogiGeneratedExperienceSource,
    run_shogi_online_replay,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run shogi Online Experience Replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--replay-sample-size", type=int, default=DEFAULT_REPLAY_SAMPLE_SIZE)
    parser.add_argument("--min-replay-size", type=int, default=DEFAULT_MIN_REPLAY_SIZE)
    parser.add_argument("--experience-store-dir", type=Path)
    parser.add_argument("--replay-seed-data-selection", type=Path)
    parser.add_argument("--training-eval-data-selection", type=Path)
    parser.add_argument("--next-checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument(
        "--experience-source",
        action="append",
        default=[],
        metavar="KIND:GAMES",
        help="Generated experience source. Repeatable. KIND is self or usi.",
    )
    parser.add_argument("--opponent", choices=("self", "usi"), default="self")
    parser.add_argument("--usi-command", help="USI engine command used when --opponent usi.")
    parser.add_argument("--usi-option", action="append", default=[], help="USI engine option as NAME=VALUE.")
    parser.add_argument("--usi-go-command", default="go nodes 1")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--concurrent-games-per-process", type=int, default=1)
    parser.add_argument("--generation-progress-every-plies", type=int, default=0)
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--generation-worker-processes", type=int, default=1)
    parser.add_argument("--mcts-move-time-limit-sec", type=float)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    result = run_shogi_online_replay(
        ShogiOnlineReplayConfig(
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            cycles=args.cycles,
            replay_capacity=args.replay_capacity,
            replay_sample_size=args.replay_sample_size,
            min_replay_size=args.min_replay_size,
            experience_store_dir=args.experience_store_dir,
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
            evaluation_batch_size=args.evaluation_batch_size,
            generation_worker_processes=args.generation_worker_processes,
            mcts_move_time_limit_sec=args.mcts_move_time_limit_sec,
            eval_ratio=args.eval_ratio,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
            device=args.device,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )
    print(json.dumps(result.to_json(), indent=2))


def _experience_sources_from_args(args: argparse.Namespace) -> tuple[ShogiGeneratedExperienceSource, ...]:
    values = args.experience_source or [f"{args.opponent}:{args.games}"]
    sources = []
    for index, value in enumerate(values):
        kind, separator, games = value.partition(":")
        if not separator:
            raise SystemExit("--experience-source must be KIND:GAMES")
        if kind not in {"self", "usi"}:
            raise SystemExit("--experience-source KIND must be self or usi")
        source = ShogiGeneratedExperienceSource(
            name=f"self-play-{index}" if kind == "self" else f"checkpoint-vs-usi-{index}",
            opponent=kind,
            games=int(games),
            usi_command=args.usi_command if kind == "usi" else None,
            usi_options=tuple(args.usi_option) if kind == "usi" else (),
            usi_go_command=args.usi_go_command,
        )
        sources.append(source)
    return tuple(sources)


if __name__ == "__main__":
    main()
