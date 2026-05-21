from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from intrep.domains.shogi.training_data_bundle import create_shogi_training_data_bundle, parse_shogi_actor_pair_ratios


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a fixed shogi training data bundle from train/eval game logs.")
    parser.add_argument("--train-games", type=Path, action="append", required=True)
    parser.add_argument("--eval-games", type=Path, required=True)
    parser.add_argument("--analysis-source", type=Path, action="append", default=[])
    parser.add_argument("--max-train-games", type=int)
    parser.add_argument("--max-eval-games", type=int)
    parser.add_argument(
        "--eval-position-policy",
        choices=("allow_overlap", "exclude_train_position_games"),
        default="allow_overlap",
    )
    parser.add_argument("--actor-pair-ratio", action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/training-data-bundles"))
    parser.add_argument(
        "--policy-target-construction",
        choices=("chosen_move", "decision_usi_multipv", "engine_analysis_multipv", "mcts_visit_counts"),
        default="chosen_move",
    )
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--policy-mate-cp", type=float, default=100000.0)
    parser.add_argument(
        "--value-target-construction",
        choices=("winner", "decision_usi_score", "engine_analysis_score"),
        default="winner",
    )
    parser.add_argument("--score-cp-scale", type=float, default=600.0)
    parser.add_argument(
        "--skip-position-stats",
        action="store_true",
        help="Skip replay-derived position statistics for large bundles.",
    )
    args = parser.parse_args(argv)

    if len(args.train_games) > 1:
        print(
            "warning: multiple --train-games inputs are intended for temporary experiments "
            "or explicit source mixes. Prefer a stable generated record set "
            "input for durable Training Data Bundles.",
            file=sys.stderr,
        )

    result = create_shogi_training_data_bundle(
        train_games=tuple(args.train_games),
        eval_games=args.eval_games,
        max_train_games=args.max_train_games,
        max_eval_games=args.max_eval_games,
        analysis_sources=tuple(args.analysis_source),
        eval_position_policy=args.eval_position_policy,
        actor_pair_ratios=parse_shogi_actor_pair_ratios(args.actor_pair_ratio),
        seed=args.seed,
        name=args.name,
        output_root=args.output_root,
        policy_target_construction=args.policy_target_construction,
        policy_temperature_cp=args.policy_temperature_cp,
        policy_mate_cp=args.policy_mate_cp,
        value_target_construction=args.value_target_construction,
        score_cp_scale=args.score_cp_scale,
        include_position_stats=not args.skip_position_stats,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
