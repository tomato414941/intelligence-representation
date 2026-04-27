from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

from intrep.mixed_corpus import MixedDocument, load_mixed_documents_jsonl


MARKERS: tuple[str, ...] = ("<obs>", "<action>", "<next_obs>")
_TOKEN_PROXY_PATTERN = re.compile(r"\S+")


@dataclass(frozen=True)
class LengthSummary:
    total_bytes: int
    total_token_proxy: int
    min_bytes: int
    max_bytes: int
    mean_bytes: float


@dataclass(frozen=True)
class ModalityCoverage:
    modality: str
    document_count: int
    length: LengthSummary


@dataclass(frozen=True)
class CorpusCoverage:
    document_count: int
    modalities: tuple[ModalityCoverage, ...]
    marker_document_counts: dict[str, int]
    length: LengthSummary


def summarize_corpus_coverage(documents: Sequence[MixedDocument]) -> CorpusCoverage:
    modality_groups: dict[str, list[MixedDocument]] = {}
    for document in documents:
        modality_groups.setdefault(document.modality, []).append(document)

    modality_summaries = tuple(
        ModalityCoverage(
            modality=modality,
            document_count=len(group),
            length=_summarize_lengths(group),
        )
        for modality, group in sorted(modality_groups.items())
    )
    return CorpusCoverage(
        document_count=len(documents),
        modalities=modality_summaries,
        marker_document_counts={
            marker: sum(1 for document in documents if marker in document.content)
            for marker in MARKERS
        },
        length=_summarize_lengths(documents),
    )


def coverage_to_dict(coverage: CorpusCoverage) -> dict[str, object]:
    return {
        "document_count": coverage.document_count,
        "modalities": [
            {
                "modality": modality.modality,
                "document_count": modality.document_count,
                "length": _length_to_dict(modality.length),
            }
            for modality in coverage.modalities
        ],
        "marker_document_counts": dict(coverage.marker_document_counts),
        "length": _length_to_dict(coverage.length),
    }


def load_corpus_coverage(path: str | Path) -> CorpusCoverage:
    return summarize_corpus_coverage(load_mixed_documents_jsonl(path))


def format_coverage_text(coverage: CorpusCoverage) -> str:
    lines = [
        f"documents\t{coverage.document_count}",
        (
            "length\t"
            f"bytes={coverage.length.total_bytes}\t"
            f"token_proxy={coverage.length.total_token_proxy}\t"
            f"mean_bytes={coverage.length.mean_bytes:.1f}"
        ),
        "markers\t"
        + "\t".join(
            f"{marker}={coverage.marker_document_counts[marker]}" for marker in MARKERS
        ),
        "modalities",
    ]
    for modality in coverage.modalities:
        lines.append(
            f"{modality.modality}\t"
            f"documents={modality.document_count}\t"
            f"bytes={modality.length.total_bytes}\t"
            f"token_proxy={modality.length.total_token_proxy}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report diagnostic coverage for MixedDocument JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        coverage = load_corpus_coverage(args.input)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(coverage_to_dict(coverage), ensure_ascii=False, sort_keys=True))
        return
    print(format_coverage_text(coverage))


def _summarize_lengths(documents: Sequence[MixedDocument]) -> LengthSummary:
    byte_lengths = [len(document.content.encode("utf-8")) for document in documents]
    token_proxy_lengths = [_token_proxy_length(document.content) for document in documents]
    if not byte_lengths:
        return LengthSummary(
            total_bytes=0,
            total_token_proxy=0,
            min_bytes=0,
            max_bytes=0,
            mean_bytes=0.0,
        )
    return LengthSummary(
        total_bytes=sum(byte_lengths),
        total_token_proxy=sum(token_proxy_lengths),
        min_bytes=min(byte_lengths),
        max_bytes=max(byte_lengths),
        mean_bytes=mean(byte_lengths),
    )


def _token_proxy_length(content: str) -> int:
    return len(_TOKEN_PROXY_PATTERN.findall(content))


def _length_to_dict(length: LengthSummary) -> dict[str, object]:
    return {
        "total_bytes": length.total_bytes,
        "total_token_proxy": length.total_token_proxy,
        "min_bytes": length.min_bytes,
        "max_bytes": length.max_bytes,
        "mean_bytes": length.mean_bytes,
    }


if __name__ == "__main__":
    main()
