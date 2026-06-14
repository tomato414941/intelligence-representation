from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.playing_strength_eval import (
    DEFAULT_MCTS_MOVE_TIME_LIMIT_SEC,
    DEFAULT_MCTS_SIMULATIONS,
    DEFAULT_NN_LEAF_EVAL_BATCH_LIMIT,
    DEFAULT_OPENING_PLIES,
    DEFAULT_SHOGI_MAX_PLIES,
    ShogiPlayingStrengthEvalConfig,
    checkpoint_playing_strength_player,
    run_shogi_playing_strength_eval,
    usi_engine_playing_strength_player,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the formal shogi playing-strength evaluation protocol.")
    parser.add_argument("--arena-repo", type=Path, default=Path("../shogi-arena-agent"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--player-a-checkpoint", type=Path, required=True)
    parser.add_argument("--player-a-checkpoint-id")
    parser.add_argument("--player-b-checkpoint", type=Path)
    parser.add_argument("--player-b-checkpoint-id")
    parser.add_argument("--player-b-usi-command")
    parser.add_argument("--player-b-usi-option", action="append", default=[])
    parser.add_argument("--player-b-usi-go-command", default="go nodes 1")
    parser.add_argument("--player-b-usi-read-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--start-position-seed", type=int, required=True)
    parser.add_argument("--opening-plies", type=int, default=DEFAULT_OPENING_PLIES)
    parser.add_argument("--max-plies", type=int, default=DEFAULT_SHOGI_MAX_PLIES)
    parser.add_argument("--simulations", type=int, default=DEFAULT_MCTS_SIMULATIONS)
    parser.add_argument("--nn-leaf-eval-batch-limit", type=int, default=DEFAULT_NN_LEAF_EVAL_BATCH_LIMIT)
    parser.add_argument("--mcts-move-time-limit-sec", type=float, default=DEFAULT_MCTS_MOVE_TIME_LIMIT_SEC)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--board-backend", choices=("python-shogi", "cshogi"), default="cshogi")
    parser.add_argument("--progress-every-games", type=int, default=1)
    args = parser.parse_args(argv)

    if (args.player_b_checkpoint is None) == (args.player_b_usi_command is None):
        parser.error("exactly one of --player-b-checkpoint or --player-b-usi-command is required")

    player_b = (
        checkpoint_playing_strength_player(
            args.player_b_checkpoint,
            checkpoint_id=args.player_b_checkpoint_id,
        )
        if args.player_b_checkpoint is not None
        else usi_engine_playing_strength_player(
            command=args.player_b_usi_command,
            options=tuple(args.player_b_usi_option),
            go_command=args.player_b_usi_go_command,
            read_timeout_seconds=args.player_b_usi_read_timeout_seconds,
        )
    )
    result = run_shogi_playing_strength_eval(
        ShogiPlayingStrengthEvalConfig(
            arena_repo=args.arena_repo,
            output_dir=args.output_dir,
            player_a=checkpoint_playing_strength_player(
                args.player_a_checkpoint,
                checkpoint_id=args.player_a_checkpoint_id,
            ),
            player_b=player_b,
            games=args.games,
            start_position_seed=args.start_position_seed,
            opening_plies=args.opening_plies,
            max_plies=args.max_plies,
            simulations=args.simulations,
            nn_leaf_eval_batch_limit=args.nn_leaf_eval_batch_limit,
            mcts_move_time_limit_sec=args.mcts_move_time_limit_sec,
            device=args.device,
            board_backend=args.board_backend,
            progress_every_games=args.progress_every_games,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
