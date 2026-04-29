from __future__ import annotations

import hashlib
import json
from pathlib import Path

from intrep.image_rendering import render_image_token_document
from intrep.mixed_corpus import MixedDocument, load_mixed_documents_jsonl
from intrep.signal_io import first_json_record, load_signals_jsonl, signal_to_record
from intrep.signals import (
    ACTION_CHANNEL,
    CONSEQUENCE_CHANNEL,
    OBSERVATION_CHANNEL,
    TEXT_CHANNEL,
    TOOL_RESULT_CHANNEL,
    Signal,
    render_payload_text,
)
from intrep.signal_stream import render_signal


MODALITY_TO_CHANNEL: dict[str, str] = {
    "text": TEXT_CHANNEL,
    "code": TEXT_CHANNEL,
    "environment_symbolic": OBSERVATION_CHANNEL,
    "environment_natural": OBSERVATION_CHANNEL,
    "external_web": OBSERVATION_CHANNEL,
    "external_action": ACTION_CHANNEL,
    "grid": OBSERVATION_CHANNEL,
    "next_grid": CONSEQUENCE_CHANNEL,
    "next_text": CONSEQUENCE_CHANNEL,
    "action_log": ACTION_CHANNEL,
    "log": TOOL_RESULT_CHANNEL,
    "tool_log": TOOL_RESULT_CHANNEL,
}


def infer_channel_from_modality(modality: str) -> str:
    if modality in MODALITY_TO_CHANNEL:
        return MODALITY_TO_CHANNEL[modality]
    if ":" in modality:
        return modality.split(":", 1)[0]
    return TEXT_CHANNEL


def mixed_document_to_signal(document: MixedDocument) -> Signal:
    return Signal(
        channel=infer_channel_from_modality(document.modality),
        payload=document.content,
    )


def mixed_documents_to_signals(documents: list[MixedDocument]) -> list[Signal]:
    return [mixed_document_to_signal(document) for document in documents]


def signal_to_mixed_document(
    event: Signal,
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
) -> MixedDocument:
    if render_format == "plain":
        return MixedDocument(
            id=_document_id(event),
            modality=event.channel,
            content=render_payload_text(event),
        )
    if render_format == "image-tokens":
        return MixedDocument(
            id=_document_id(event),
            modality=event.channel,
            content=render_image_token_document(
                event,
                patch_size=image_patch_size,
                channel_bins=image_channel_bins,
                token_format=image_token_format,
            ),
        )
    if render_format not in ("signal-tags", "typed-tags"):
        raise ValueError("render_format must be plain, signal-tags, typed-tags, or image-tokens")
    return MixedDocument(
        id=_document_id(event),
        modality=event.channel,
        content=render_signal(event),
    )


def signals_to_mixed_documents(
    events: list[Signal],
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
) -> list[MixedDocument]:
    return [
        signal_to_mixed_document(
            event,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
            image_token_format=image_token_format,
        )
        for event in events
    ]


def load_signals_as_mixed_documents(
    path: str | Path,
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
) -> list[MixedDocument]:
    return signals_to_mixed_documents(
        load_signals_jsonl(path),
        render_format=render_format,
        image_patch_size=image_patch_size,
        image_channel_bins=image_channel_bins,
        image_token_format=image_token_format,
    )


def load_corpus_jsonl_as_mixed_documents(
    path: str | Path,
    *,
    corpus_format: str = "auto",
    render_format: str = "plain",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
) -> list[MixedDocument]:
    if corpus_format == "mixed-document":
        documents = load_mixed_documents_jsonl(path)
        return _render_mixed_documents_if_requested(documents, render_format=render_format)
    if corpus_format in ("signal", "typed-event"):
        return load_signals_as_mixed_documents(
            path,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
            image_token_format=image_token_format,
        )
    if corpus_format != "auto":
        raise ValueError("corpus_format must be auto, mixed-document, signal, or typed-event")

    first_record = first_json_record(path)
    if isinstance(first_record, dict) and ("channel" in first_record or "role" in first_record):
        return load_signals_as_mixed_documents(
            path,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
            image_token_format=image_token_format,
        )
    documents = load_mixed_documents_jsonl(path)
    return _render_mixed_documents_if_requested(documents, render_format=render_format)


def _render_mixed_documents_if_requested(
    documents: list[MixedDocument],
    *,
    render_format: str,
) -> list[MixedDocument]:
    if render_format == "plain":
        return documents
    if render_format in ("signal-tags", "typed-tags", "image-tokens"):
        return signals_to_mixed_documents(mixed_documents_to_signals(documents))
    raise ValueError("render_format must be plain, signal-tags, typed-tags, or image-tokens")


def _document_id(event: Signal) -> str:
    identity = json.dumps(signal_to_record(event), ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{event.channel}_{digest}"
