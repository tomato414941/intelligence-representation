from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.problems.shogi_policy_value.tensor_cache import build_shogi_policy_value_tensor_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tensor cache for a shogi policy/value data selection.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--shard-games", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    summary = build_shogi_policy_value_tensor_cache(
        data_selection_path=args.data_selection,
        output_path=args.out,
        shard_games=args.shard_games,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
