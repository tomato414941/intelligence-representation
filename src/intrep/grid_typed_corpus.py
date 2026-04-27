from __future__ import annotations

from collections.abc import Iterable

from intrep.grid_corpus import default_grid_documents
from intrep.mixed_corpus import MixedDocument
from intrep.typed_corpus import mixed_document_to_typed_event
from intrep.typed_events import EventRole, TypedEvent


def default_grid_typed_events() -> list[TypedEvent]:
    return grid_documents_to_typed_events(default_grid_documents())


def grid_documents_to_typed_events(documents: Iterable[MixedDocument]) -> list[TypedEvent]:
    events: list[TypedEvent] = []
    for document in documents:
        event = mixed_document_to_typed_event(document)
        step_id, time_index = _grid_step_metadata(document)
        if step_id is not None:
            event = TypedEvent(
                id=document.id,
                role=_grid_role(document.modality),
                modality=document.modality,
                content=document.content,
                episode_id=step_id,
                time_index=time_index,
                source_id=document.id,
                metadata=event.metadata,
            )
        events.append(event)
    return events


def _grid_step_metadata(document: MixedDocument) -> tuple[str | None, int | None]:
    suffix_times = {
        "_grid": 0,
        "_action_log": 1,
        "_next_grid": 2,
        "_next_text": 3,
        "_text": -1,
    }
    for suffix, time_index in suffix_times.items():
        if document.id.endswith(suffix):
            return document.id[: -len(suffix)], time_index
    return None, None


def _grid_role(modality: str) -> EventRole:
    if modality == "action_log":
        return EventRole.ACTION
    if modality in ("next_grid", "next_text"):
        return EventRole.CONSEQUENCE
    if modality in ("grid", "text"):
        return EventRole.OBSERVATION
    return EventRole.TEXT
