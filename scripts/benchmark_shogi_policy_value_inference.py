from __future__ import annotations

import argparse
from pathlib import Path

from intrep.problems.shogi_policy_value.benchmarking import (
    benchmark_shogi_policy_value_inference_batching,
    load_position_sfens_from_jsonl,
    parse_batch_sizes,
    write_json_result,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark shogi policy/value checkpoint inference batching.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positions-jsonl", type=Path, required=True)
    parser.add_argument("--sfen-field", default="position_sfen")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    position_sfens = load_position_sfens_from_jsonl(
        args.positions_jsonl,
        sfen_field=args.sfen_field,
        limit=args.limit,
    )
    result = benchmark_shogi_policy_value_inference_batching(
        args.checkpoint,
        position_sfens,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        device=args.device,
        dtype=args.dtype,
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
    )
    write_json_result(result, args.out)


if __name__ == "__main__":
    main()
