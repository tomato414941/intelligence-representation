from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.worlds.shogi.source_selection import create_shogi_training_view_from_sources, parse_shogi_actor_pair_ratios


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a fixed shogi training view through source selection.")
    parser.add_argument("--train-games", type=Path, action="append", required=True)
    parser.add_argument("--eval-games", type=Path, required=True)
    parser.add_argument("--max-train-games", type=int)
    parser.add_argument("--max-eval-games", type=int)
    parser.add_argument("--actor-pair-ratio", action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/datasets"))
    parser.add_argument("--policy-target-source", choices=("chosen_move", "usi_multipv"), default="chosen_move")
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--policy-mate-cp", type=float, default=100000.0)
    parser.add_argument("--value-target-source", choices=("winner", "yaneuraou_best_score"), default="winner")
    parser.add_argument("--score-cp-scale", type=float, default=600.0)
    args = parser.parse_args(argv)

    result = create_shogi_training_view_from_sources(
        train_games=tuple(args.train_games),
        eval_games=args.eval_games,
        name=args.name,
        output_root=args.output_root,
        max_train_games=args.max_train_games,
        max_eval_games=args.max_eval_games,
        actor_pair_ratios=parse_shogi_actor_pair_ratios(args.actor_pair_ratio),
        seed=args.seed,
        policy_target_source=args.policy_target_source,
        policy_temperature_cp=args.policy_temperature_cp,
        policy_mate_cp=args.policy_mate_cp,
        value_target_source=args.value_target_source,
        score_cp_scale=args.score_cp_scale,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
