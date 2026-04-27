from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from intrep.mixed_corpus import (
    MixedDocument,
    load_mixed_documents_jsonl,
    write_mixed_documents_jsonl,
)


@dataclass(frozen=True)
class CorpusSplit:
    train_documents: list[MixedDocument]
    eval_documents: list[MixedDocument]


def split_mixed_documents(
    documents: list[MixedDocument],
    *,
    eval_ratio: float = 0.2,
    key: str = "id",
    seed: str = "intrep",
) -> CorpusSplit:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if key not in ("id", "modality", "content"):
        raise ValueError("key must be one of: id, modality, content")

    train_documents: list[MixedDocument] = []
    eval_documents: list[MixedDocument] = []
    threshold = int(eval_ratio * 10_000)
    for document in documents:
        value = getattr(document, key)
        bucket = _stable_bucket(f"{seed}:{value}")
        if bucket < threshold:
            eval_documents.append(document)
        else:
            train_documents.append(document)

    if documents and not train_documents:
        train_documents.append(eval_documents.pop())
    if len(documents) > 1 and not eval_documents:
        eval_documents.append(train_documents.pop())
    return CorpusSplit(train_documents=train_documents, eval_documents=eval_documents)


def write_split_jsonl(
    input_path: str | Path,
    train_path: str | Path,
    eval_path: str | Path,
    *,
    eval_ratio: float = 0.2,
    key: str = "id",
    seed: str = "intrep",
) -> CorpusSplit:
    documents = load_mixed_documents_jsonl(input_path)
    split = split_mixed_documents(documents, eval_ratio=eval_ratio, key=key, seed=seed)
    write_mixed_documents_jsonl(train_path, split.train_documents)
    write_mixed_documents_jsonl(eval_path, split.eval_documents)
    return split


def _stable_bucket(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split MixedDocument JSONL into train/eval files.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument(
        "--key",
        choices=("id", "modality", "content"),
        default="id",
        help="Stable split key. Use modality only for coarse stress tests.",
    )
    parser.add_argument("--seed", default="intrep")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        split = write_split_jsonl(
            args.input,
            args.train_output,
            args.eval_output,
            eval_ratio=args.eval_ratio,
            key=args.key,
            seed=args.seed,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        f"wrote train={len(split.train_documents)} eval={len(split.eval_documents)} "
        f"from {args.input}"
    )


if __name__ == "__main__":
    main()
