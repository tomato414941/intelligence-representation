from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.worlds.shogi.training_data_bundle import create_shogi_training_data_bundle, parse_shogi_actor_pair_ratios


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a fixed shogi training data bundle from train/eval game logs.")
    parser.add_argument("--train-games", type=Path, action="append", required=True)
    parser.add_argument("--eval-games", type=Path, required=True)
    parser.add_argument("--max-train-games", type=int)
    parser.add_argument("--max-eval-games", type=int)
    parser.add_argument("--actor-pair-ratio", action="append", default=[])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/training-data-bundles"))
    parser.add_argument("--policy-target-construction", choices=("chosen_move", "decision_usi_multipv"), default="chosen_move")
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--policy-mate-cp", type=float, default=100000.0)
    parser.add_argument("--value-target-construction", choices=("winner", "decision_usi_score"), default="winner")
    parser.add_argument("--score-cp-scale", type=float, default=600.0)
    args = parser.parse_args(argv)

    result = create_shogi_training_data_bundle(
        train_games=tuple(args.train_games),
        eval_games=args.eval_games,
        max_train_games=args.max_train_games,
        max_eval_games=args.max_eval_games,
        actor_pair_ratios=parse_shogi_actor_pair_ratios(args.actor_pair_ratio),
        seed=args.seed,
        name=args.name,
        output_root=args.output_root,
        policy_target_construction=args.policy_target_construction,
        policy_temperature_cp=args.policy_temperature_cp,
        policy_mate_cp=args.policy_mate_cp,
        value_target_construction=args.value_target_construction,
        score_cp_scale=args.score_cp_scale,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
