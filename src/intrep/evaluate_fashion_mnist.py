from __future__ import annotations

import argparse
from pathlib import Path

from intrep.evaluate_future_prediction import (
    FuturePredictionEvaluationConfig,
    run_future_prediction_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Fashion-MNIST image-to-label prediction over signal streams."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--image-channel-bins", type=int, default=4)
    parser.add_argument("--max-negatives", type=int, default=3)
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_future_prediction_evaluation(
        FuturePredictionEvaluationConfig(
            train_path=args.train_path,
            eval_path=args.eval_path,
            target_channel="label",
            max_steps=args.max_steps,
            context_length=args.context_length,
            batch_size=args.batch_size,
            batch_stride=args.batch_stride,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            rendering="image-tokens",
            image_patch_size=args.image_patch_size,
            image_channel_bins=args.image_channel_bins,
            max_negatives=args.max_negatives,
            metrics_path=args.metrics_path,
        )
    )


if __name__ == "__main__":
    main()
