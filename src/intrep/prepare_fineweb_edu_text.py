from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DEFAULT_TEXT_COLUMN = "text"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a local text slice from FineWeb-Edu.")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--max-documents", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_streaming_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
    )
    result = write_text_slice(
        records=records,
        output_path=args.output_path,
        max_bytes=args.max_bytes,
        text_column=args.text_column,
        max_documents=args.max_documents,
    )
    print("intrep prepare fineweb edu text")
    print(
        f"dataset={args.dataset_name}"
        f" split={args.split}"
        f" documents={result.document_count}"
        f" bytes={result.byte_count}"
        f" output_path={args.output_path}"
    )


@dataclass(frozen=True)
class TextSliceResult:
    document_count: int
    byte_count: int


def load_streaming_dataset(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Preparing FineWeb-Edu text requires the Hugging Face datasets package. "
            "Install it with `uv pip install datasets` or an equivalent environment command."
        ) from error

    if dataset_config is None:
        return load_dataset(dataset_name, split=split, streaming=True)
    return load_dataset(dataset_name, dataset_config, split=split, streaming=True)


def write_text_slice(
    *,
    records: Iterable[Mapping[str, Any]],
    output_path: Path,
    max_bytes: int,
    text_column: str = DEFAULT_TEXT_COLUMN,
    max_documents: int | None = None,
) -> TextSliceResult:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive when provided")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    documents = 0
    with output_path.open("wb") as handle:
        for record in records:
            if max_documents is not None and documents >= max_documents:
                break
            text = record.get(text_column)
            if not isinstance(text, str) or not text.strip():
                continue
            chunk = (text.strip() + "\n\n").encode("utf-8")
            remaining = max_bytes - written
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = _truncate_utf8(chunk, remaining)
                if not chunk:
                    break
            handle.write(chunk)
            written += len(chunk)
            documents += 1
            if written >= max_bytes:
                break

    return TextSliceResult(document_count=documents, byte_count=written)


def _truncate_utf8(payload: bytes, max_bytes: int) -> bytes:
    return payload[:max_bytes].decode("utf-8", errors="ignore").encode("utf-8")


def _run_as_script() -> None:
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hugging Face streaming over parquet can leave pyarrow finalizers in a bad
    # state after intentionally stopping early at the byte limit. The corpus has
    # already been written and flushed, so bypass interpreter shutdown finalizers
    # for this CLI path only.
    os._exit(0)


if __name__ == "__main__":
    _run_as_script()
