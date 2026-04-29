from __future__ import annotations

import json
from pathlib import Path

from intrep.signals import PayloadRef, Signal


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
        events.append(signal_from_record(record, line_number=line_number))
    return events


def load_signals_jsonl_v2(path: str | Path) -> list[Signal]:
    return load_signals_jsonl(path)


def write_signals_jsonl_v2(path: str | Path, events: list[Signal]) -> None:
    lines = [json.dumps(signal_to_record(event), ensure_ascii=False) for event in events]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def reject_payload_refs(events: list[Signal], *, context: str) -> None:
    if any(isinstance(event.payload, PayloadRef) for event in events):
        raise ValueError(
            f"{context} does not support payload_ref; "
            "requires a channel-specific loader or encoder"
        )


def first_json_record(path: str | Path) -> object:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return None


def signal_from_record(record: object, *, line_number: int) -> Signal:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
    extra_fields = set(record.keys()) - {
        "channel",
        "payload",
        "payload_ref",
        "id",
        "episode_id",
        "time_index",
    }
    if extra_fields:
        fields = ", ".join(sorted(extra_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: unsupported fields: {fields}"
        )
    missing_fields = set()
    if "payload" not in record and "payload_ref" not in record:
        missing_fields.add("payload")
    if "channel" not in record:
        missing_fields.add("channel")
    if missing_fields:
        fields = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
        )
    if "payload" in record and "payload_ref" in record:
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: "
            "payload and payload_ref are mutually exclusive"
        )
    payload: str | PayloadRef
    if "payload_ref" in record:
        payload = payload_ref_from_record(record["payload_ref"], line_number=line_number)
    else:
        payload = record["payload"]
        if not isinstance(payload, str):
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: field payload must be a string"
            )
    channel = record["channel"]
    if not isinstance(channel, str):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field channel must be a string"
        )
    return Signal(channel=channel, payload=payload)


def signal_to_record(event: Signal) -> dict[str, object]:
    if isinstance(event.payload, PayloadRef):
        return {
            "channel": event.channel,
            "payload_ref": payload_ref_to_record(event.payload),
        }
    return {
        "channel": event.channel,
        "payload": event.payload,
    }


def payload_ref_to_record(ref: PayloadRef) -> dict[str, object]:
    record: dict[str, object] = {
        "uri": ref.uri,
        "media_type": ref.media_type,
    }
    if ref.sha256 is not None:
        record["sha256"] = ref.sha256
    if ref.size_bytes is not None:
        record["size_bytes"] = ref.size_bytes
    return record


def payload_ref_from_record(record: object, *, line_number: int) -> PayloadRef:
    if not isinstance(record, dict):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: field payload_ref must be an object"
        )
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
