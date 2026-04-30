from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.image_classification import load_image_choice_examples_jsonl
from intrep.text_image_shared_core_training import (
    TextImageSharedCoreTrainingConfig,
    train_text_image_shared_core_with_result,
)
from intrep.train_language_model import read_text_corpora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train text language modeling and image classification through one shared Transformer core."
    )
    parser.add_argument("--corpus-path", type=Path, action="append", required=True)
    parser.add_argument("--image-train-path", type=Path, required=True)
    parser.add_argument("--image-eval-path", type=Path)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--text-context-length", type=int, default=64)
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    text_corpus = read_text_corpora(args.corpus_path, seed=args.seed)
    train_examples = load_image_choice_examples_jsonl(args.image_train_path)
    eval_examples = (
        load_image_choice_examples_jsonl(args.image_eval_path)
        if args.image_eval_path is not None
        else None
    )
    result = train_text_image_shared_core_with_result(
        text_corpus=text_corpus,
        image_train_examples=train_examples,
        image_eval_examples=eval_examples,
        config=TextImageSharedCoreTrainingConfig(
            text_context_length=args.text_context_length,
            patch_size=args.image_patch_size,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
            tokenizer_vocab_size=args.tokenizer_vocab_size,
        ),
    )
    payload = {
        "schema_version": "intrep.text_image_shared_core_run.v1",
        "corpus_paths": [str(path) for path in args.corpus_path],
        "image_train_path": str(args.image_train_path),
        "image_eval_path": str(args.image_eval_path) if args.image_eval_path is not None else None,
        "metrics": result.metrics.to_dict(),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("intrep text image shared core")
    print(
        f"shared_core={result.metrics.shared_core}"
        f" text_tokens={result.metrics.text_token_count}"
        f" image_train_cases={result.metrics.image_train_case_count}"
        f" text_initial_loss={result.metrics.text_initial_loss:.4f}"
        f" text_final_loss={result.metrics.text_final_loss:.4f}"
        f" image_initial_loss={result.metrics.image_initial_loss:.4f}"
        f" image_final_loss={result.metrics.image_final_loss:.4f}"
        f" image_train_accuracy={result.metrics.image_train_accuracy:.4f}"
        f" image_eval_accuracy={result.metrics.image_eval_accuracy if result.metrics.image_eval_accuracy is not None else 'none'}"
    )


if __name__ == "__main__":
    main()
