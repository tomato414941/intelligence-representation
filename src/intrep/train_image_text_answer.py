from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.image_text_answer_training import (
    ImageTextAnswerTrainingConfig,
    load_image_text_answer_examples_jsonl,
    train_image_text_answer_model,
)
from intrep.image_text_answer_checkpoint import save_image_text_answer_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-to-text answer model from image-text-answer JSONL.")
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--tokenizer-corpus-path", type=Path)
    parser.add_argument("--metrics-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--text-context-length", type=int, default=32)
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_examples = load_image_text_answer_examples_jsonl(args.train_path)
    result = train_image_text_answer_model(
        train_examples=train_examples,
        tokenizer_corpus=_read_optional_text(args.tokenizer_corpus_path),
        config=ImageTextAnswerTrainingConfig(
            text_context_length=args.text_context_length,
            image_patch_size=args.image_patch_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
        ),
    )
    if args.metrics_path is not None:
        args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_path.write_text(
            json.dumps(
                {
                    "schema_version": "intrep.image_text_answer_run.v1",
                    "train_path": str(args.train_path),
                    "tokenizer_corpus_path": (
                        str(args.tokenizer_corpus_path) if args.tokenizer_corpus_path is not None else None
                    ),
                    "metrics": asdict(result.metrics),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.checkpoint_path is not None:
        save_image_text_answer_checkpoint(args.checkpoint_path, result)
    print("intrep image text answer")
    print(
        f"train_cases={result.metrics.train_case_count}"
        f" initial_loss={result.metrics.train_initial_loss:.4f}"
        f" final_loss={result.metrics.train_final_loss:.4f}"
        f" max_steps={result.metrics.max_steps}"
        f" model_preset={result.metrics.model_preset}"
    )


def _read_optional_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
