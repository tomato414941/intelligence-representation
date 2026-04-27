from __future__ import annotations

import argparse
from pathlib import Path

from intrep import train_gpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PTM typed-event path using the current decoder-only GPT backend."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="small")
    parser.add_argument("--loss-summary", action="store_true")
    parser.add_argument("--run-summary-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    forwarded = [
        "--corpus",
        "file",
        "--corpus-path",
        str(args.train_path),
        "--corpus-format",
        "typed-event",
        "--render-format",
        "typed-tags",
        "--max-steps",
        str(args.max_steps),
        "--context-length",
        str(args.context_length),
        "--batch-size",
        str(args.batch_size),
        "--model-preset",
        args.model_preset,
    ]
    if args.eval_path is not None:
        forwarded.extend(["--eval-corpus-path", str(args.eval_path)])
    if args.batch_stride is not None:
        forwarded.extend(["--batch-stride", str(args.batch_stride)])
    if args.loss_summary:
        forwarded.append("--loss-summary")
    if args.run_summary_path is not None:
        forwarded.extend(["--run-summary-path", str(args.run_summary_path)])
    train_gpt.main(forwarded)


if __name__ == "__main__":
    main()
