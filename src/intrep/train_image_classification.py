from __future__ import annotations

import argparse
from pathlib import Path

from intrep.image_classification import (
    ImageClassificationConfig,
    ImageFolderClassificationDataset,
    load_image_classification_examples_jsonl,
    train_image_classifier_with_result,
    write_metrics,
)
from intrep.image_classification_checkpoint import save_image_classification_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train image classification with image patch embeddings and the shared Transformer core."
    )
    train_input = parser.add_mutually_exclusive_group(required=True)
    train_input.add_argument("--train-path", type=Path)
    train_input.add_argument("--train-image-folder", type=Path)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--eval-image-folder", type=Path)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.eval_path is not None and args.eval_image_folder is not None:
        raise SystemExit("provide only one of --eval-path or --eval-image-folder")
    image_size = tuple(args.image_size) if args.image_size is not None else None
    train_examples = None
    train_dataset = None
    if args.train_path is not None:
        train_examples = load_image_classification_examples_jsonl(args.train_path)
    else:
        train_dataset = ImageFolderClassificationDataset(args.train_image_folder, image_size=image_size)
    eval_examples = None
    eval_dataset = None
    if args.eval_path is not None:
        eval_examples = load_image_classification_examples_jsonl(args.eval_path)
    elif args.eval_image_folder is not None:
        eval_dataset = ImageFolderClassificationDataset(args.eval_image_folder, image_size=image_size)
    result = train_image_classifier_with_result(
        train_examples=train_examples,
        eval_examples=eval_examples,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=ImageClassificationConfig(
            patch_size=args.image_patch_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
        ),
    )
    metrics = result.metrics
    if args.metrics_path is not None:
        write_metrics(args.metrics_path, metrics)
    if args.checkpoint_path is not None:
        save_image_classification_checkpoint(args.checkpoint_path, result)
    print("intrep image classification")
    print(
        f"target={metrics.target}"
        f" input_representation={metrics.input_representation}"
        f" train_cases={metrics.train_case_count}"
        f" eval_cases={metrics.eval_case_count}"
        f" train_initial_loss={metrics.train_initial_loss:.4f}"
        f" train_final_loss={metrics.train_final_loss:.4f}"
        f" train_accuracy={metrics.train_accuracy:.4f}"
        f" eval_accuracy={metrics.eval_accuracy if metrics.eval_accuracy is not None else 'none'}"
    )


if __name__ == "__main__":
    main()
