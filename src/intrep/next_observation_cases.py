from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from intrep.mixed_corpus import MixedDocument


@dataclass(frozen=True)
class NextObservationCase:
    id: str
    modality: str
    prefix: str
    positive_next: str
    hard_negative_nexts: tuple[str, ...] = field(default_factory=tuple)
    group_id: str | None = None


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
        modality = _base_modality(document.modality)
        if modality not in ("environment_symbolic", "external_action", "tool_log"):
            continue
        parsed = _split_symbolic_next_observation(document.content)
        if parsed is None:
            continue
        prefix, positive_next, hard_negative_nexts, group_id = parsed
        cases.append(
            NextObservationCase(
                id=document.id,
                modality=modality,
                prefix=prefix,
                positive_next=positive_next,
                hard_negative_nexts=hard_negative_nexts,
                group_id=group_id,
            )
        )
    return cases


def _extract_grid_cases(documents: Sequence[MixedDocument]) -> list[NextObservationCase]:
    by_id = {document.id: document for document in documents}
    cases: list[NextObservationCase] = []

    for document in documents:
        if _base_modality(document.modality) != "grid" or not document.id.endswith("_grid"):
            continue
        step_id = document.id[: -len("_grid")]
        action = by_id.get(f"{step_id}_action_log")
        next_grid = by_id.get(f"{step_id}_next_grid")
        if action is None or _base_modality(action.modality) != "action_log":
            continue
        if next_grid is None or _base_modality(next_grid.modality) != "next_grid":
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


def _base_modality(modality: str) -> str:
    return modality.split(":", 1)[1] if ":" in modality else modality


def _split_symbolic_next_observation(content: str) -> tuple[str, str, tuple[str, ...], str | None] | None:
    obs_match = _find_marker(content, "obs")
    action_match = _find_marker(content, "action")
    next_match = _find_marker(content, "next_obs")
    if obs_match is None or action_match is None or next_match is None:
        return None
    if not (obs_match.start() < next_match.start()):
        return None
    if action_match.start() > next_match.start():
        return None

    positive_next = _strip_event_closing_tag(content[next_match.end() :].strip())
    if not positive_next:
        return None

    metadata = _stable_marker_metadata(content)
    prefix = f"{content[: next_match.start()].rstrip()} {next_match.group(0)} "
    return (
        prefix,
        positive_next,
        metadata.get("hard_negative_nexts", ()),
        metadata.get("group_id"),
    )


def _strip_event_closing_tag(value: str) -> str:
    return value[: -len("</EVENT>")].rstrip() if value.endswith("</EVENT>") else value


def _find_marker(content: str, marker_name: str) -> re.Match[str] | None:
    return re.search(rf"<{re.escape(marker_name)}(?:\s+[^>]*)?>", content)


def _stable_marker_metadata(content: str) -> dict[str, str | tuple[str, ...]]:
    metadata: dict[str, str | tuple[str, ...]] = {}
    for marker_name in ("case", "next_observation_case", "obs", "next_obs"):
        marker = _find_marker(content, marker_name)
        if marker is None:
            continue
        attributes = _parse_marker_attributes(marker.group(0))
        group_id = attributes.get("group_id") or attributes.get("entity_id")
        if group_id and "group_id" not in metadata:
            metadata["group_id"] = group_id
        hard_negative_nexts = _split_hard_negative_nexts(attributes)
        if hard_negative_nexts and "hard_negative_nexts" not in metadata:
            metadata["hard_negative_nexts"] = hard_negative_nexts
    return metadata


def _parse_marker_attributes(marker: str) -> dict[str, str]:
    return {
        match.group("name"): match.group("quoted") or match.group("bare")
        for match in re.finditer(
            r"(?P<name>[A-Za-z_][A-Za-z0-9_-]*)="
            r"(?:\"(?P<quoted>[^\"]*)\"|(?P<bare>[^\s>]+))",
            marker,
        )
    }


def _split_hard_negative_nexts(attributes: dict[str, str]) -> tuple[str, ...]:
    value = attributes.get("hard_negative_nexts") or attributes.get("hard_negative_next")
    if value is None:
        return ()
    return tuple(part.strip() for part in re.split(r"\s*\|\|?\s*", value) if part.strip())
