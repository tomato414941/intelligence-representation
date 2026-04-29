from __future__ import annotations

import argparse
from pathlib import Path

from intrep.fashion_mnist_vit import (
    ImageClassificationConfig,
    train_fashion_mnist_classifier,
    write_metrics,
)
from intrep.signal_corpus import load_signals_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Fashion-MNIST image classification with a patch Transformer."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_events = load_signals_jsonl(args.train_path)
    eval_events = load_signals_jsonl(args.eval_path) if args.eval_path is not None else None
    metrics = train_fashion_mnist_classifier(
        train_events=train_events,
        eval_events=eval_events,
        config=ImageClassificationConfig(
            patch_size=args.image_patch_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
        ),
    )
    if args.metrics_path is not None:
        write_metrics(args.metrics_path, metrics)
    print("intrep fashion-mnist vit")
    print(
        f"target_channel={metrics.target_channel}"
        f" rendering={metrics.rendering}"
        f" train_cases={metrics.train_case_count}"
        f" eval_cases={metrics.eval_case_count}"
        f" train_initial_loss={metrics.train_initial_loss:.4f}"
        f" train_final_loss={metrics.train_final_loss:.4f}"
        f" train_accuracy={metrics.train_accuracy:.4f}"
        f" eval_accuracy={metrics.eval_accuracy if metrics.eval_accuracy is not None else 'none'}"
    )


if __name__ == "__main__":
    main()
