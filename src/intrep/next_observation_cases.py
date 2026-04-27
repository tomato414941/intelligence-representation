from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intrep.mixed_corpus import MixedDocument


@dataclass(frozen=True)
class NextObservationCase:
    id: str
    modality: str
    prefix: str
    positive_next: str


def extract_next_observation_cases(
    documents: Sequence[MixedDocument],
) -> list[NextObservationCase]:
    cases: list[NextObservationCase] = []
    cases.extend(_extract_marker_next_observation_cases(documents))
    cases.extend(_extract_grid_cases(documents))
    return cases


def _extract_marker_next_observation_cases(
    documents: Sequence[MixedDocument],
) -> list[NextObservationCase]:
    cases: list[NextObservationCase] = []
    for document in documents:
        if document.modality not in ("environment_symbolic", "external_action"):
            continue
        parsed = _split_symbolic_next_observation(document.content)
        if parsed is None:
            continue
        prefix, positive_next = parsed
        cases.append(
            NextObservationCase(
                id=document.id,
                modality=document.modality,
                prefix=prefix,
                positive_next=positive_next,
            )
        )
    return cases


def _extract_grid_cases(documents: Sequence[MixedDocument]) -> list[NextObservationCase]:
    by_id = {document.id: document for document in documents}
    cases: list[NextObservationCase] = []

    for document in documents:
        if document.modality != "grid" or not document.id.endswith("_grid"):
            continue
        step_id = document.id[: -len("_grid")]
        action = by_id.get(f"{step_id}_action_log")
        next_grid = by_id.get(f"{step_id}_next_grid")
        if action is None or action.modality != "action_log":
            continue
        if next_grid is None or next_grid.modality != "next_grid":
            continue

        cases.append(
            NextObservationCase(
                id=step_id,
                modality="grid",
                prefix=f"{document.content}\n{action.content}\n",
                positive_next=next_grid.content,
            )
        )

    return cases


def _split_symbolic_next_observation(content: str) -> tuple[str, str] | None:
    obs_marker = "<obs>"
    action_marker = "<action>"
    next_marker = "<next_obs>"
    obs_index = content.find(obs_marker)
    action_index = content.find(action_marker)
    next_index = content.find(next_marker)
    if not (0 <= obs_index < action_index < next_index):
        return None

    before_next, positive_next = content.split(next_marker, 1)
    positive_next = positive_next.strip()
    if not positive_next:
        return None
    return f"{before_next.rstrip()} {next_marker} ", positive_next
