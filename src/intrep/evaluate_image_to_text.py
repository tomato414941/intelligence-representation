from __future__ import annotations

import argparse
from pathlib import Path

from intrep.image_classification import load_image_choice_examples_jsonl
from intrep.image_to_text_training import (
    ImageToTextTrainingConfig,
    train_image_to_text_labels_with_result,
    write_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train image-conditioned text label output with image patch embeddings and token loss."
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
    parser.add_argument("--choice-eval-limit", type=int, default=200)
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_examples = load_image_choice_examples_jsonl(args.train_path)
    eval_examples = load_image_choice_examples_jsonl(args.eval_path) if args.eval_path is not None else None
    result = train_image_to_text_labels_with_result(
        train_examples=train_examples,
        eval_examples=eval_examples,
        config=ImageToTextTrainingConfig(
            patch_size=args.image_patch_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
            choice_eval_limit=args.choice_eval_limit,
        ),
    )
    metrics = result.metrics
    if args.metrics_path is not None:
        write_metrics(args.metrics_path, metrics)
    print("intrep image to text")
    print(
        f"target={metrics.target}"
        f" input_representation={metrics.input_representation}"
        f" output_representation={metrics.output_representation}"
        f" train_cases={metrics.train_case_count}"
        f" eval_cases={metrics.eval_case_count}"
        f" train_initial_loss={metrics.train_initial_loss:.4f}"
        f" train_final_loss={metrics.train_final_loss:.4f}"
        f" eval_final_loss={metrics.eval_final_loss if metrics.eval_final_loss is not None else 'none'}"
        f" train_choice_accuracy={metrics.train_choice_accuracy if metrics.train_choice_accuracy is not None else 'none'}"
        f" train_choice_cases={metrics.train_choice_case_count}"
        f" eval_choice_accuracy={metrics.eval_choice_accuracy if metrics.eval_choice_accuracy is not None else 'none'}"
        f" eval_choice_cases={metrics.eval_choice_case_count}"
    )


if __name__ == "__main__":
    main()
