from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypeAlias


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class ShogiPositionIndexEntry:
    kind: str
    source: str
    created_at: str | None = None
    settings: dict[str, JsonValue] = field(default_factory=dict)
    data: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("kind must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")


@dataclass(frozen=True)
class ShogiPositionIndexRecord:
    position_sfen: str
    entries: tuple[ShogiPositionIndexEntry, ...]

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        if not self.entries:
            raise ValueError("entries must not be empty")


class ShogiPositionIndex:
    def __init__(self, records: list[ShogiPositionIndexRecord]) -> None:
        by_position: dict[str, ShogiPositionIndexRecord] = {}
        for record in records:
            if record.position_sfen in by_position:
                raise ValueError(f"duplicate shogi position index record: {record.position_sfen}")
            by_position[record.position_sfen] = record
        self._by_position = by_position

    def get(self, position_sfen: str) -> ShogiPositionIndexRecord | None:
        return self._by_position.get(position_sfen)

    def entries(
        self,
        position_sfen: str,
        *,
        kind: str | None = None,
        source: str | None = None,
    ) -> tuple[ShogiPositionIndexEntry, ...]:
        record = self.get(position_sfen)
        if record is None:
            return ()
        return tuple(
            entry
            for entry in record.entries
            if (kind is None or entry.kind == kind) and (source is None or entry.source == source)
        )


def write_shogi_position_index_jsonl(path: str | Path, records: list[ShogiPositionIndexRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(shogi_position_index_record_to_json(record), separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_shogi_position_index_jsonl(path: str | Path) -> ShogiPositionIndex:
    return ShogiPositionIndex(list(iter_shogi_position_index_jsonl(path)))


def iter_shogi_position_index_jsonl(path: str | Path) -> Iterator[ShogiPositionIndexRecord]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                yield shogi_position_index_record_from_json(json.loads(stripped))


def shogi_position_index_record_to_json(record: ShogiPositionIndexRecord) -> dict[str, object]:
    return {
        "position_sfen": record.position_sfen,
        "entries": [shogi_position_index_entry_to_json(entry) for entry in record.entries],
    }


def shogi_position_index_record_from_json(payload: dict[str, object]) -> ShogiPositionIndexRecord:
    return ShogiPositionIndexRecord(
        position_sfen=str(payload["position_sfen"]),
        entries=tuple(
            shogi_position_index_entry_from_json(_object_dict(entry))
            for entry in _object_list(payload.get("entries", []))
        ),
    )


def shogi_position_index_entry_to_json(entry: ShogiPositionIndexEntry) -> dict[str, object]:
    return {
        "kind": entry.kind,
        "source": entry.source,
        "created_at": entry.created_at,
        "settings": entry.settings,
        "data": entry.data,
    }


def shogi_position_index_entry_from_json(payload: dict[str, object]) -> ShogiPositionIndexEntry:
    return ShogiPositionIndexEntry(
        kind=str(payload["kind"]),
        source=str(payload["source"]),
        created_at=None if payload.get("created_at") is None else str(payload["created_at"]),
        settings=_json_object(payload.get("settings", {})),
        data=_json_object(payload.get("data", {})),
    )


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


def _json_object(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise ValueError(f"unsupported json value: {type(value).__name__}")
