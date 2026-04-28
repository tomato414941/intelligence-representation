from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from intrep.mixed_corpus import MixedDocument
from intrep.signals import (
    ACTION_CHANNEL,
    CONSEQUENCE_CHANNEL,
    OBSERVATION_CHANNEL,
    TEXT_CHANNEL,
    TOOL_RESULT_CHANNEL,
    PayloadRef,
    Signal,
    render_payload_text,
)
from intrep.signal_stream import render_signal, render_signal_stream


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


def render_signal_corpus(events: list[Signal]) -> str:
    return render_signal_stream(events)


def signal_to_mixed_document(
    event: Signal,
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
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
            content=_render_image_token_document(
                event,
                patch_size=image_patch_size,
                channel_bins=image_channel_bins,
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
) -> list[MixedDocument]:
    return [
        signal_to_mixed_document(
            event,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
        )
        for event in events
    ]


def load_signals_jsonl(path: str | Path) -> list[Signal]:
    events: list[Signal] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
        events.append(_signal_from_record(record, line_number=line_number))
    return events


def load_signals_jsonl_v2(path: str | Path) -> list[Signal]:
    events: list[Signal] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
        missing_fields = set()
        if "payload" not in record and "payload_ref" not in record and "content" not in record:
            missing_fields.add("payload")
        if "channel" not in record and "role" not in record and "modality" not in record:
            missing_fields.add("channel")
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
            )
        events.append(_signal_from_record(record, line_number=line_number))
    return events


def write_signals_jsonl_v2(path: str | Path, events: list[Signal]) -> None:
    lines = [json.dumps(_signal_to_record(event), ensure_ascii=False) for event in events]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def reject_payload_refs(events: list[Signal], *, context: str) -> None:
    if any(isinstance(event.payload, PayloadRef) for event in events):
        raise ValueError(
            f"{context} does not support payload_ref; "
            "requires a channel-specific loader or encoder"
        )


def load_signals_as_mixed_documents(
    path: str | Path,
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
) -> list[MixedDocument]:
    return signals_to_mixed_documents(
        load_signals_jsonl(path),
        render_format=render_format,
        image_patch_size=image_patch_size,
        image_channel_bins=image_channel_bins,
    )


def load_corpus_jsonl_as_mixed_documents(
    path: str | Path,
    *,
    corpus_format: str = "auto",
    render_format: str = "plain",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
) -> list[MixedDocument]:
    from intrep.mixed_corpus import load_mixed_documents_jsonl

    if corpus_format == "mixed-document":
        documents = load_mixed_documents_jsonl(path)
        return _render_mixed_documents_if_requested(documents, render_format=render_format)
    if corpus_format in ("signal", "typed-event"):
        return load_signals_as_mixed_documents(
            path,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
        )
    if corpus_format != "auto":
        raise ValueError("corpus_format must be auto, mixed-document, signal, or typed-event")

    first_record = _first_json_record(path)
    if isinstance(first_record, dict) and ("channel" in first_record or "role" in first_record):
        return load_signals_as_mixed_documents(
            path,
            render_format=render_format,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
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


def _render_image_token_document(
    event: Signal,
    *,
    patch_size: int,
    channel_bins: int,
) -> str:
    if event.channel != "image":
        return render_payload_text(event)
    if not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)

    from intrep.image_tokenizer import ImagePatchTokenizer

    tokenizer = ImagePatchTokenizer(patch_size=patch_size, channel_bins=channel_bins)
    token_ids = tokenizer.encode_ref(event.payload)
    return (
        f"<IMAGE_TOKENS patch_size=\"{tokenizer.patch_size}\" "
        f"channel_bins=\"{tokenizer.channel_bins}\">\n"
        + " ".join(str(token_id) for token_id in token_ids)
        + "\n</IMAGE_TOKENS>\n"
    )


def _first_json_record(path: str | Path) -> Any:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return None


def _signal_from_record(record: object, *, line_number: int) -> Signal:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
    missing_fields = set()
    if "payload" not in record and "payload_ref" not in record and "content" not in record:
        missing_fields.add("payload")
    if "channel" not in record and "role" not in record and "modality" not in record:
        missing_fields.add("channel")
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
        )
    if "payload" in record and "payload_ref" in record:
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: payload and payload_ref are mutually exclusive"
        )
    payload: str | PayloadRef
    if "payload_ref" in record:
        payload = _payload_ref_from_record(record["payload_ref"], line_number=line_number)
    else:
        payload = record.get("payload", record.get("content"))
        if not isinstance(payload, str):
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: field payload must be a string"
            )
    channel = record.get("channel", record.get("role", record.get("modality")))
    if not isinstance(channel, str):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field channel must be a string"
        )
    return Signal(
        channel=channel,
        payload=payload,
    )


def _document_id(event: Signal) -> str:
    identity = json.dumps(
        _signal_to_record(event),
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{event.channel}_{digest}"


def _signal_to_record(event: Signal) -> dict[str, object]:
    if isinstance(event.payload, PayloadRef):
        return {
            "channel": event.channel,
            "payload_ref": _payload_ref_to_record(event.payload),
        }
    return {
        "channel": event.channel,
        "payload": event.payload,
    }


def _payload_ref_to_record(ref: PayloadRef) -> dict[str, object]:
    record: dict[str, object] = {
        "uri": ref.uri,
        "media_type": ref.media_type,
    }
    if ref.sha256 is not None:
        record["sha256"] = ref.sha256
    if ref.size_bytes is not None:
        record["size_bytes"] = ref.size_bytes
    return record


def _payload_ref_from_record(record: object, *, line_number: int) -> PayloadRef:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: field payload_ref must be an object")
    missing_fields = {"uri", "media_type"} - record.keys()
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: missing payload_ref fields: {fields}"
        )
    extra_fields = set(record.keys()) - {"uri", "media_type", "sha256", "size_bytes"}
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: unsupported payload_ref fields: {fields}"
        )
    try:
        return PayloadRef(
            uri=record["uri"],
            media_type=record["media_type"],
            sha256=record.get("sha256"),
            size_bytes=record.get("size_bytes"),
        )
    except ValueError as error:
        raise ValueError(f"Invalid JSONL record at line {line_number}: {error}") from error
