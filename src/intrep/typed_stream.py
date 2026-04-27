from __future__ import annotations

from collections.abc import Sequence

from intrep.mixed_corpus import MixedDocument
from intrep.typed_events import EventRole, TypedEvent, render_typed_event


def render_typed_stream(events: Sequence[TypedEvent]) -> str:
    return "\n".join(render_typed_event(event) for event in _stable_event_order(events))


def mixed_document_to_typed_event(document: MixedDocument) -> TypedEvent:
    return TypedEvent(
        id=document.id,
        role=EventRole.TEXT,
        modality=document.modality,
        content=document.content,
    )


def render_mixed_documents_as_typed_stream(documents: Sequence[MixedDocument]) -> str:
    return render_typed_stream([mixed_document_to_typed_event(document) for document in documents])


def _stable_event_order(events: Sequence[TypedEvent]) -> list[TypedEvent]:
    return sorted(
        events,
        key=lambda event: (
            event.episode_id or "",
            event.time_index if event.time_index is not None else 10**12,
            event.id,
        ),
    )
