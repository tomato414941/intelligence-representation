from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.image_classification import load_image_choice_examples_jsonl
from intrep.image_to_text_training import evaluate_image_to_text_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an image-to-text model checkpoint on image-choice examples.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    examples = load_image_choice_examples_jsonl(args.eval_path)
    metrics = evaluate_image_to_text_checkpoint(
        checkpoint_path=args.checkpoint_path,
        examples=examples,
        device=args.device,
    )
    if args.metrics_path is not None:
        args.metrics_path.write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("intrep image to text checkpoint")
    print(f"cases={metrics.case_count} accuracy={metrics.accuracy:.4f}")


if __name__ == "__main__":
    main()
