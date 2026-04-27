from __future__ import annotations

import math
from dataclasses import dataclass

from intrep.mixed_corpus import MixedDocument


ENV_SYMBOLIC_MODALITY = "environment_symbolic"
ENV_NATURAL_MODALITY = "environment_natural"


@dataclass(frozen=True)
class MixedCorpusPairingCoverage:
    modality_counts: dict[str, int]
    environment_symbolic_count: int
    environment_natural_count: int
    paired_episode_ids: list[str]


@dataclass(frozen=True)
class MixedEnvironmentDocumentPair:
    episode_id: str
    symbolic: MixedDocument
    natural: MixedDocument


@dataclass(frozen=True)
class MixedCorpusDocumentSplit:
    train_documents: list[MixedDocument]
    eval_documents: list[MixedDocument]
    eval_episode_ids: list[str]


def evaluate_mixed_corpus_pairing(
    documents: list[MixedDocument],
) -> MixedCorpusPairingCoverage:
    modality_counts: dict[str, int] = {}

    for document in documents:
        modality_counts[document.modality] = modality_counts.get(document.modality, 0) + 1
    pairs = extract_environment_document_pairs(documents)

    return MixedCorpusPairingCoverage(
        modality_counts=modality_counts,
        environment_symbolic_count=modality_counts.get(ENV_SYMBOLIC_MODALITY, 0),
        environment_natural_count=modality_counts.get(ENV_NATURAL_MODALITY, 0),
        paired_episode_ids=[pair.episode_id for pair in pairs],
    )


def extract_environment_document_pairs(
    documents: list[MixedDocument],
) -> list[MixedEnvironmentDocumentPair]:
    symbolic_by_episode_id: dict[str, MixedDocument] = {}
    natural_by_episode_id: dict[str, MixedDocument] = {}

    for document in documents:
        episode_id = _environment_episode_id(document)
        if episode_id is None:
            continue
        if document.modality == ENV_SYMBOLIC_MODALITY:
            symbolic_by_episode_id.setdefault(episode_id, document)
        elif document.modality == ENV_NATURAL_MODALITY:
            natural_by_episode_id.setdefault(episode_id, document)

    return [
        MixedEnvironmentDocumentPair(
            episode_id=episode_id,
            symbolic=symbolic_by_episode_id[episode_id],
            natural=natural_by_episode_id[episode_id],
        )
        for episode_id in sorted(symbolic_by_episode_id.keys() & natural_by_episode_id.keys())
    ]


def build_train_eval_document_split(
    documents: list[MixedDocument],
    *,
    eval_episode_count: int | None = None,
    eval_episode_fraction: float | None = None,
) -> MixedCorpusDocumentSplit:
    if (eval_episode_count is None) == (eval_episode_fraction is None):
        raise ValueError("provide exactly one of eval_episode_count or eval_episode_fraction")

    paired_episode_ids = [pair.episode_id for pair in extract_environment_document_pairs(documents)]
    if eval_episode_count is None:
        eval_episode_count = _eval_count_from_fraction(
            len(paired_episode_ids),
            eval_episode_fraction,
        )
    elif eval_episode_count < 0:
        raise ValueError("eval_episode_count must be non-negative")

    if eval_episode_count > len(paired_episode_ids):
        raise ValueError("eval_episode_count exceeds paired environment episode count")

    eval_episode_ids = paired_episode_ids[:eval_episode_count]
    eval_episode_id_set = set(eval_episode_ids)
    train_documents: list[MixedDocument] = []
    eval_documents: list[MixedDocument] = []

    for document in documents:
        episode_id = _environment_episode_id(document)
        if episode_id in eval_episode_id_set:
            eval_documents.append(document)
        else:
            train_documents.append(document)

    return MixedCorpusDocumentSplit(
        train_documents=train_documents,
        eval_documents=eval_documents,
        eval_episode_ids=eval_episode_ids,
    )


def _eval_count_from_fraction(paired_episode_count: int, fraction: float | None) -> int:
    if fraction is None:
        raise ValueError("eval_episode_fraction is required")
    if not 0 <= fraction <= 1:
        raise ValueError("eval_episode_fraction must be between 0 and 1")
    return math.ceil(paired_episode_count * fraction)


def _environment_episode_id(document: MixedDocument) -> str | None:
    prefixes = {
        ENV_SYMBOLIC_MODALITY: (
            ("env_symbolic_", ""),
            ("env_pair_symbolic_", "pair_"),
        ),
        ENV_NATURAL_MODALITY: (
            ("env_natural_", ""),
            ("env_pair_natural_", "pair_"),
        ),
    }
    modality_prefixes = prefixes.get(document.modality)
    if modality_prefixes is None:
        return None
    for prefix, episode_prefix in modality_prefixes:
        if document.id.startswith(prefix):
            episode_id = document.id.removeprefix(prefix)
            return f"{episode_prefix}{episode_id}" if episode_id else None
    return None
