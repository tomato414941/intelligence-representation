from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.fashion_mnist_vit import ImageClassificationConfig, load_image_choice_examples_jsonl
from intrep.image_text_training import ImageTextTrainingConfig
from intrep.language_modeling_training import LanguageModelingTrainingConfig
from intrep.text_examples import LanguageModelingExample
from intrep.training_phases import ImageTextTrainingPhasesResult, run_image_text_training_phases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run staged image classification, text modeling, and image-text training."
    )
    parser.add_argument("--image-examples-path", type=Path, required=True)
    parser.add_argument("--text-corpus-path", type=Path, required=True)
    parser.add_argument("--prompt", default="label: ")
    parser.add_argument("--image-max-steps", type=int, default=20)
    parser.add_argument("--text-max-steps", type=int, default=20)
    parser.add_argument("--image-text-max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    image_examples = load_image_choice_examples_jsonl(args.image_examples_path)
    result = run_image_text_training_phases(
        image_examples=image_examples,
        text_examples=_read_text_examples(args.text_corpus_path),
        image_classification_config=ImageClassificationConfig(
            patch_size=args.image_patch_size,
            max_steps=args.image_max_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_preset=args.model_preset,
            device=args.device,
        ),
        language_modeling_config=LanguageModelingTrainingConfig(
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_steps=args.text_max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            tokenizer="byte",
        ),
        image_text_config=ImageTextTrainingConfig(
            max_steps=args.image_text_max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        prompt=args.prompt,
    )
    payload = training_phases_result_to_dict(result)
    if args.metrics_path is not None:
        args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("intrep training phases")
    print(
        f"image_cases={result.image_classification.metrics.train_case_count}"
        f" text_tokens={result.language_modeling.result.token_count}"
        f" image_text_cases={result.image_text.case_count}"
        f" image_text_initial_loss={result.image_text.initial_loss:.4f}"
        f" image_text_final_loss={result.image_text.final_loss:.4f}"
        f" image_text_choice_accuracy={result.image_text_choice.accuracy:.4f}"
    )


def training_phases_result_to_dict(result: ImageTextTrainingPhasesResult) -> dict[str, object]:
    return {
        "image_classification": result.image_classification.metrics.to_dict(),
        "language_modeling": asdict(result.language_modeling.result),
        "image_text": asdict(result.image_text),
        "image_text_choice": result.image_text_choice.to_dict(),
    }


def _read_text_examples(path: Path) -> tuple[LanguageModelingExample, ...]:
    text = path.read_text(encoding="utf-8")
    if not text:
        raise ValueError("text corpus must not be empty")
    return (LanguageModelingExample(text),)


if __name__ == "__main__":
    main()
