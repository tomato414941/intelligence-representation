from __future__ import annotations

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


def evaluate_mixed_corpus_pairing(
    documents: list[MixedDocument],
) -> MixedCorpusPairingCoverage:
    modality_counts: dict[str, int] = {}
    symbolic_episode_ids: set[str] = set()
    natural_episode_ids: set[str] = set()

    for document in documents:
        modality_counts[document.modality] = modality_counts.get(document.modality, 0) + 1
        episode_id = _environment_episode_id(document)
        if episode_id is None:
            continue
        if document.modality == ENV_SYMBOLIC_MODALITY:
            symbolic_episode_ids.add(episode_id)
        elif document.modality == ENV_NATURAL_MODALITY:
            natural_episode_ids.add(episode_id)

    return MixedCorpusPairingCoverage(
        modality_counts=modality_counts,
        environment_symbolic_count=modality_counts.get(ENV_SYMBOLIC_MODALITY, 0),
        environment_natural_count=modality_counts.get(ENV_NATURAL_MODALITY, 0),
        paired_episode_ids=sorted(symbolic_episode_ids & natural_episode_ids),
    )


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
