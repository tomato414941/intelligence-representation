from __future__ import annotations

import argparse
from pathlib import Path

from intrep.text_tokenizer import TextTokenizerKind, build_text_tokenizer, save_text_tokenizer
from intrep.train_language_model import read_text_corpora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and save a text tokenizer.")
    parser.add_argument("--corpus-path", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--tokenizer", choices=("byte", "byte-pair"), default="byte-pair")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-min-pair-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    corpus = read_text_corpora(args.corpus_path, seed=args.seed)
    tokenizer = build_text_tokenizer(
        corpus,
        kind=args.tokenizer,
        vocab_size=args.tokenizer_vocab_size,
        min_pair_count=args.tokenizer_min_pair_count,
    )
    save_text_tokenizer(args.tokenizer_path, tokenizer)
    print("intrep train text tokenizer")
    print(
        f"corpus_paths={len(args.corpus_path)}"
        f" tokenizer={args.tokenizer}"
        f" vocab_size={tokenizer.vocab_size}"
    )


if __name__ == "__main__":
    main()
