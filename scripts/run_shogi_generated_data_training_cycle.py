from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.generated_data_cycle import (
    DEFAULT_SHOGI_MAX_PLIES,
    ShogiGeneratedDataTrainingCycleConfig,
    run_shogi_generated_data_training_cycle,
)
from intrep.problems.shogi_policy_value.generated_game_production import (
    DEFAULT_USI_READ_TIMEOUT_SECONDS,
    checkpoint_generated_player,
    usi_engine_generated_player,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one generated shogi data training cycle.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument("--black-player", choices=("checkpoint", "usi-engine"), default="checkpoint")
    parser.add_argument("--white-player", choices=("checkpoint", "usi-engine"), default="checkpoint")
    parser.add_argument("--usi-command", help="USI engine command used by usi-engine players.")
    parser.add_argument("--usi-option", action="append", default=[], help="USI engine option as NAME=VALUE.")
    parser.add_argument("--usi-go-command", default="go nodes 1")
    parser.add_argument("--usi-read-timeout-seconds", type=float, default=DEFAULT_USI_READ_TIMEOUT_SECONDS)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--concurrent-games-per-process", type=int, default=1)
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=1)
    parser.add_argument("--generation-worker-processes", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mcts-move-time-limit-sec", type=float)
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--allow-nonstandard-loss-weights", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args(argv)

    result = run_shogi_generated_data_training_cycle(
        ShogiGeneratedDataTrainingCycleConfig(
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            arena_repo=args.arena_repo,
            black_player=_player_from_args(args, side="black"),
            white_player=_player_from_args(args, side="white"),
            games=args.games,
            concurrent_games_per_process=args.concurrent_games_per_process,
            board_backend=args.board_backend,
            max_plies=args.max_plies,
            simulations=args.simulations,
            evaluation_batch_size=args.evaluation_batch_size,
            generation_worker_processes=args.generation_worker_processes,
            seed=args.seed,
            mcts_move_time_limit_sec=args.mcts_move_time_limit_sec,
            eval_ratio=args.eval_ratio,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            policy_loss_weight=args.policy_loss_weight,
            value_loss_weight=args.value_loss_weight,
            allow_nonstandard_loss_weights=args.allow_nonstandard_loss_weights,
            device=args.device,
            num_workers=args.num_workers,
        )
    )
    print(json.dumps(result.to_json(), indent=2))


def _player_from_args(args: argparse.Namespace, *, side: str):
    kind = getattr(args, f"{side}_player")
    if kind == "checkpoint":
        return checkpoint_generated_player(side)
    if not args.usi_command:
        raise SystemExit("--usi-command is required when a player is usi-engine")
    return usi_engine_generated_player(
        name=f"{side}-usi-engine",
        command=args.usi_command,
        options=tuple(args.usi_option),
        go_command=args.usi_go_command,
        read_timeout_seconds=args.usi_read_timeout_seconds,
    )


if __name__ == "__main__":
    main()
