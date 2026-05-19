from __future__ import annotations

import argparse
from pathlib import Path

from intrep.problems.shogi_policy_value.benchmarking import (
    benchmark_shogi_position_feature_generation,
    load_position_sfens_from_jsonl,
    write_json_result,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark shogi position feature generation latency.")
    parser.add_argument("--positions-jsonl", type=Path, required=True)
    parser.add_argument("--sfen-field", default="position_sfen")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    position_sfens = load_position_sfens_from_jsonl(
        args.positions_jsonl,
        sfen_field=args.sfen_field,
        limit=args.limit,
    )
    result = benchmark_shogi_position_feature_generation(
        position_sfens,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    write_json_result(result, args.out)


if __name__ == "__main__":
    main()
