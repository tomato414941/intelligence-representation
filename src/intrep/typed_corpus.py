from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from intrep.mixed_corpus import MixedDocument
from intrep.typed_events import EVENT_ROLES, EventRole, TypedEvent
from intrep.typed_stream import render_typed_event, render_typed_stream


MODALITY_TO_ROLE: dict[str, EventRole] = {
    "text": "text",
    "code": "text",
    "environment_symbolic": "observation",
    "environment_natural": "observation",
    "external_web": "observation",
    "external_action": "action",
    "grid": "observation",
    "next_grid": "consequence",
    "next_text": "consequence",
    "action_log": "action",
    "log": "tool_result",
    "tool_log": "tool_result",
}


def infer_role_from_modality(modality: str) -> EventRole:
    if modality in MODALITY_TO_ROLE:
        return MODALITY_TO_ROLE[modality]
    if ":" in modality:
        prefix = modality.split(":", 1)[0]
        if prefix in EVENT_ROLES:
            return cast(EventRole, prefix)
    return "text"


def mixed_document_to_typed_event(document: MixedDocument) -> TypedEvent:
    return TypedEvent(
        id=document.id,
        role=infer_role_from_modality(document.modality),
        modality=document.modality,
        content=document.content,
        source_id=document.id,
    )


def mixed_documents_to_typed_events(documents: list[MixedDocument]) -> list[TypedEvent]:
    return [mixed_document_to_typed_event(document) for document in documents]


def render_typed_corpus(events: list[TypedEvent]) -> str:
    return render_typed_stream(events)


def typed_event_to_mixed_document(event: TypedEvent, *, render_format: str = "typed-tags") -> MixedDocument:
    if render_format == "plain":
        return MixedDocument(id=event.id, modality=event.modality, content=event.content)
    if render_format != "typed-tags":
        raise ValueError("render_format must be plain or typed-tags")
    return MixedDocument(
        id=event.id,
        modality=f"{event.role}:{event.modality}",
        content=render_typed_event(event),
    )


def typed_events_to_mixed_documents(
    events: list[TypedEvent],
    *,
    render_format: str = "typed-tags",
) -> list[MixedDocument]:
    return [typed_event_to_mixed_document(event, render_format=render_format) for event in events]


def load_typed_events_jsonl(path: str | Path) -> list[TypedEvent]:
    events: list[TypedEvent] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
        events.append(_typed_event_from_record(record, line_number=line_number))
    return events


def load_typed_events_jsonl_v2(path: str | Path) -> list[TypedEvent]:
    events: list[TypedEvent] = []
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
        missing_fields = {
            "id",
            "role",
            "modality",
            "content",
            "episode_id",
            "time_index",
            "metadata",
        } - record.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
            )
        events.append(_typed_event_from_record(record, line_number=line_number))
    return events


def write_typed_events_jsonl_v2(path: str | Path, events: list[TypedEvent]) -> None:
    lines = [
        json.dumps(
            {
                "id": event.id,
                "role": event.role,
                "modality": event.modality,
                "content": event.content,
                "episode_id": event.episode_id,
                "time_index": event.time_index,
                "metadata": dict(event.metadata),
            },
            ensure_ascii=False,
        )
        for event in events
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_typed_events_as_mixed_documents(
    path: str | Path,
    *,
    render_format: str = "typed-tags",
) -> list[MixedDocument]:
    return typed_events_to_mixed_documents(
        load_typed_events_jsonl(path),
        render_format=render_format,
    )


def load_corpus_jsonl_as_mixed_documents(
    path: str | Path,
    *,
    corpus_format: str = "auto",
    render_format: str = "plain",
) -> list[MixedDocument]:
    from intrep.mixed_corpus import load_mixed_documents_jsonl

    if corpus_format == "mixed-document":
        documents = load_mixed_documents_jsonl(path)
        return _render_mixed_documents_if_requested(documents, render_format=render_format)
    if corpus_format == "typed-event":
        return load_typed_events_as_mixed_documents(path, render_format=render_format)
    if corpus_format != "auto":
        raise ValueError("corpus_format must be auto, mixed-document, or typed-event")

    first_record = _first_json_record(path)
    if isinstance(first_record, dict) and "role" in first_record:
        return load_typed_events_as_mixed_documents(path, render_format=render_format)
    documents = load_mixed_documents_jsonl(path)
    return _render_mixed_documents_if_requested(documents, render_format=render_format)


def _render_mixed_documents_if_requested(
    documents: list[MixedDocument],
    *,
    render_format: str,
) -> list[MixedDocument]:
    if render_format == "plain":
        return documents
    if render_format == "typed-tags":
        return typed_events_to_mixed_documents(mixed_documents_to_typed_events(documents))
    raise ValueError("render_format must be plain or typed-tags")


def _first_json_record(path: str | Path) -> Any:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return None


def _typed_event_from_record(record: object, *, line_number: int) -> TypedEvent:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
    missing_fields = {"id", "role", "modality", "content"} - record.keys()
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
        )
    for field in ("id", "role", "modality", "content"):
        if not isinstance(record[field], str):
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: field {field} must be a string"
            )
    if record["role"] not in EVENT_ROLES:
        raise ValueError(f"Invalid JSONL record at line {line_number}: unknown role {record['role']}")
    time_index = record.get("time_index")
    if time_index is not None and not isinstance(time_index, int):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field time_index must be an int"
        )
    episode_id = record.get("episode_id")
    if episode_id is not None and not isinstance(episode_id, str):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field episode_id must be a string"
        )
    source_id = record.get("source_id")
    if source_id is not None and not isinstance(source_id, str):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field source_id must be a string"
        )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field metadata must be an object"
        )
    return TypedEvent(
        id=record["id"],
        role=cast(EventRole, record["role"]),
        modality=record["modality"],
        content=record["content"],
        time_index=time_index,
        episode_id=episode_id,
        source_id=source_id,
        metadata=metadata,
    )
