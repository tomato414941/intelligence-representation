from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.image_classification import load_image_text_choice_examples_jsonl
from intrep.image_text_choice_checkpoint import save_image_text_choice_checkpoint
from intrep.image_text_choice_training import (
    ImageTextChoiceTrainingConfig,
    train_image_text_choice_model,
)
from intrep.shared_multimodal_checkpoint import load_shared_multimodal_initialization
from intrep.text_tokenizer import load_text_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-text choice model from image-text-choice JSONL.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--tokenizer-corpus-path", type=Path)
    parser.add_argument("--language-modeling-corpus-path", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--init-checkpoint-path", type=Path)
    parser.add_argument("--text-context-length", type=int, default=16)
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.tokenizer_path is not None and args.tokenizer_corpus_path is not None:
        raise SystemExit("provide only one of --tokenizer-path or --tokenizer-corpus-path")
    train_examples = load_image_text_choice_examples_jsonl(args.train_path)
    eval_examples = (
        load_image_text_choice_examples_jsonl(args.eval_path)
        if args.eval_path is not None
        else None
    )
    initialization = (
        load_shared_multimodal_initialization(args.init_checkpoint_path, device=args.device)
        if args.init_checkpoint_path is not None
        else None
    )
    tokenizer = (
        load_text_tokenizer(args.tokenizer_path)
        if args.tokenizer_path is not None
        else initialization.tokenizer if initialization is not None else None
    )
    result = train_image_text_choice_model(
        train_examples=train_examples,
        eval_examples=eval_examples,
        tokenizer_corpus=_read_optional_text(args.tokenizer_corpus_path),
        language_modeling_corpus=_read_optional_text(args.language_modeling_corpus_path) or None,
        prompt=args.prompt,
        config=ImageTextChoiceTrainingConfig(
            text_context_length=args.text_context_length,
            image_patch_size=args.image_patch_size,
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
            tokenizer_vocab_size=args.tokenizer_vocab_size,
        ),
        tokenizer_override=tokenizer,
        initial_model_state_dict=(
            initialization.model_state_dict if initialization is not None else None
        ),
    )
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(
            json.dumps(
                {
                    "schema_version": "intrep.image_text_choice_run.v1",
                    "train_path": str(args.train_path),
                    "eval_path": str(args.eval_path) if args.eval_path is not None else None,
                    "tokenizer_corpus_path": (
                        str(args.tokenizer_corpus_path) if args.tokenizer_corpus_path is not None else None
                    ),
                    "tokenizer_path": str(args.tokenizer_path) if args.tokenizer_path is not None else None,
                    "language_modeling_corpus_path": (
                        str(args.language_modeling_corpus_path)
                        if args.language_modeling_corpus_path is not None
                        else None
                    ),
                    "prompt": args.prompt,
                    "init_checkpoint_path": (
                        str(args.init_checkpoint_path) if args.init_checkpoint_path is not None else None
                    ),
                    "init_checkpoint_schema": initialization.source_schema if initialization is not None else None,
                    "metrics": asdict(result.metrics),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.checkpoint_path is not None:
        save_image_text_choice_checkpoint(args.checkpoint_path, result)
    print("intrep image text choice")
    print(
        f"train_cases={result.metrics.train_case_count}"
        f" eval_cases={result.metrics.eval_case_count}"
        f" initial_loss={result.metrics.train_initial_loss:.4f}"
        f" final_loss={result.metrics.train_final_loss:.4f}"
        f" train_accuracy={result.metrics.train_accuracy:.4f}"
        f" eval_accuracy={result.metrics.eval_accuracy if result.metrics.eval_accuracy is not None else 'none'}"
        f" max_steps={result.metrics.max_steps}"
        f" model_preset={result.metrics.model_preset}"
    )


def _read_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
