from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from intrep.causal_text_model import build_causal_text_config
from intrep.language_modeling_metrics import language_modeling_metrics_from_training_result
from intrep.language_modeling_training import (
    LanguageModelingTrainingConfig,
    train_language_modeling_with_artifacts,
)
from intrep.text_examples import LanguageModelingExample
from intrep.text_tokenizer import (
    TextTokenizerKind,
    build_text_tokenizer,
    load_text_tokenizer,
    text_tokenizer_to_payload,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a causal text model on a text corpus.")
    parser.add_argument("--corpus-path", type=Path, action="append", required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--tokenizer", choices=("byte", "byte-pair", "hf-byte-pair"), default="byte")
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-min-pair-count", type=int, default=2)
    parser.add_argument("--eval-batch-limit", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_text, eval_text = read_split_text_corpora(
        args.corpus_path,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    training_config = LanguageModelingTrainingConfig(
        context_length=args.context_length,
        batch_stride=args.batch_stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        tokenizer=args.tokenizer,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_min_pair_count=args.tokenizer_min_pair_count,
        eval_batch_limit=args.eval_batch_limit,
    )
    tokenizer = _tokenizer_override(args, train_text)
    vocab_size = tokenizer.vocab_size if tokenizer is not None else _vocab_size(
        args.tokenizer,
        args.tokenizer_vocab_size,
    )
    artifacts = train_language_modeling_with_artifacts(
        train_examples=(LanguageModelingExample(train_text),),
        eval_examples=(LanguageModelingExample(eval_text),),
        training_config=training_config,
        model_config=build_causal_text_config(
            preset=args.model_preset,
            vocab_size=vocab_size,
            context_length=args.context_length,
        ),
        tokenizer_override=tokenizer,
    )
    payload = {
        "schema_version": "intrep.language_model_run.v1",
        "corpus_paths": [str(path) for path in args.corpus_path],
        "train_char_count": len(train_text),
        "eval_char_count": len(eval_text),
        "model_preset": args.model_preset,
        "training_config": _training_config_payload(training_config),
        "tokenizer": {
            "source": str(args.tokenizer_path) if args.tokenizer_path is not None else "trained",
            "payload": text_tokenizer_to_payload(artifacts.tokenizer),
        },
        "result": asdict(artifacts.result),
        "metrics": language_modeling_metrics_from_training_result(artifacts.result),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("intrep train language model")
    print(
        f"train_chars={len(train_text)}"
        f" eval_chars={len(eval_text)}"
        f" token_count={artifacts.result.token_count}"
        f" initial_train_loss={artifacts.result.initial_train_loss:.4f}"
        f" final_train_loss={artifacts.result.final_train_loss:.4f}"
        f" initial_eval_loss={artifacts.result.initial_eval_loss:.4f}"
        f" final_eval_loss={artifacts.result.final_eval_loss:.4f}"
    )


def read_split_text_corpora(
    corpus_paths: list[Path],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[str, str]:
    if not corpus_paths:
        raise ValueError("at least one corpus path is required")
    train_texts = []
    eval_texts = []
    for path in corpus_paths:
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"corpus must not be empty: {path}")
        train_text, eval_text = split_text_corpus(text, eval_ratio=eval_ratio)
        train_texts.append(train_text.strip())
        eval_texts.append(eval_text.strip())
    randomizer = random.Random(seed)
    randomizer.shuffle(train_texts)
    randomizer.shuffle(eval_texts)
    return join_text_documents(train_texts), join_text_documents(eval_texts)


def read_text_corpora(corpus_paths: list[Path], *, seed: int) -> str:
    if not corpus_paths:
        raise ValueError("at least one corpus path is required")
    texts = []
    for path in corpus_paths:
        text = path.read_text(encoding="utf-8")
        if not text:
            raise ValueError(f"corpus must not be empty: {path}")
        texts.append(text.strip())
    random.Random(seed).shuffle(texts)
    return join_text_documents(texts)


def join_text_documents(texts: list[str]) -> str:
    if not texts:
        raise ValueError("at least one text document is required")
    return "\n<|endoftext|>\n".join(texts) + "\n"


def split_text_corpus(text: str, *, eval_ratio: float) -> tuple[str, str]:
    if not text:
        raise ValueError("corpus must not be empty")
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    split_index = int(len(text) * (1.0 - eval_ratio))
    train_text = text[:split_index]
    eval_text = text[split_index:]
    if not train_text or not eval_text:
        raise ValueError("eval_ratio produced an empty train or eval split")
    return train_text, eval_text


def _vocab_size(tokenizer: TextTokenizerKind, tokenizer_vocab_size: int) -> int:
    if tokenizer == "byte":
        return 257
    return tokenizer_vocab_size


def _tokenizer_override(args: argparse.Namespace, train_text: str):
    if args.tokenizer_path is not None:
        return load_text_tokenizer(args.tokenizer_path)
    if args.tokenizer == "hf-byte-pair":
        return build_text_tokenizer(
            train_text,
            kind=args.tokenizer,
            vocab_size=args.tokenizer_vocab_size,
            min_pair_count=args.tokenizer_min_pair_count,
        )
    return None


def _training_config_payload(config: LanguageModelingTrainingConfig) -> dict[str, object]:
    payload = asdict(config)
    if config.checkpoint_path is not None:
        payload["checkpoint_path"] = str(config.checkpoint_path)
    return payload


if __name__ == "__main__":
    main()
