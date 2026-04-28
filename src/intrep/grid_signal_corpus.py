from __future__ import annotations

from collections.abc import Iterable

from intrep.grid_corpus import default_grid_documents
from intrep.mixed_corpus import MixedDocument
from intrep.signals import Signal
from intrep.signal_corpus import mixed_document_to_signal


def default_grid_signals() -> list[Signal]:
    return grid_documents_to_signals(default_grid_documents())


def grid_documents_to_signals(documents: Iterable[MixedDocument]) -> list[Signal]:
    events: list[Signal] = []
    for document in documents:
        event = mixed_document_to_signal(document)
        if _grid_step_metadata(document)[0] is not None:
            event = Signal(
                channel=_grid_channel(document.modality),
                payload=document.content,
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


def _grid_channel(modality: str) -> str:
    if modality == "action_log":
        return "action"
    if modality in ("next_grid", "next_text"):
        return "consequence"
    if modality in ("grid", "text"):
        return "observation"
    return "text"
