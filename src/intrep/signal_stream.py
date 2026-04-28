from __future__ import annotations

from collections.abc import Sequence

from intrep.mixed_corpus import MixedDocument
from intrep.signals import TEXT_CHANNEL, Signal, render_signal as render_signal_tag


def render_signal_stream(events: Sequence[Signal]) -> str:
    return "\n".join(render_signal(event) for event in events)


def render_signal(event: Signal) -> str:
    return render_signal_tag(event)


def mixed_document_to_signal(document: MixedDocument) -> Signal:
    return Signal(
        channel=document.modality or TEXT_CHANNEL,
        payload=document.content,
    )


def render_mixed_documents_as_signal_stream(documents: Sequence[MixedDocument]) -> str:
    return render_signal_stream([mixed_document_to_signal(document) for document in documents])
