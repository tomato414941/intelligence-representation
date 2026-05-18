from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import shogi


@dataclass(frozen=True)
class ShogiActorSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiDecisionTelemetry:
    move_performance: dict[str, object] | None = None
    batch_performance: dict[str, object] | None = None
    search_evidence: dict[str, object] | None = None


@dataclass(frozen=True)
class ShogiMoveRecord:
    action_usi: str
    decision_usi_info_lines: tuple[str, ...] = ()
    decision_telemetry: ShogiDecisionTelemetry | None = None


@dataclass(frozen=True)
class ShogiGameRecord:
    black_actor: ShogiActorSpec
    white_actor: ShogiActorSpec
    initial_position_sfen: str
    moves: tuple[ShogiMoveRecord, ...]
    winner: str | None = None
    end_reason: str | None = None
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)


def load_shogi_game_records_jsonl(path: str | Path) -> list[ShogiGameRecord]:
    records = list(iter_shogi_game_records_jsonl(path))
    if not records:
        raise ValueError("shogi game records jsonl must contain at least one game")
    return records


def iter_shogi_game_records_jsonl(path: str | Path) -> Iterator[ShogiGameRecord]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = shogi_game_record_from_json(json.loads(stripped))
            if record.moves:
                yield record


def write_shogi_game_records_jsonl(path: str | Path, records: Sequence[ShogiGameRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if not record.moves:
            continue
        lines.append(json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True))
    if not lines:
        raise ValueError("records must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def shogi_game_record_to_json(record: ShogiGameRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "intrep.shogi_game_record.v2",
        "black_actor": shogi_actor_spec_to_json(record.black_actor),
        "white_actor": shogi_actor_spec_to_json(record.white_actor),
        "initial_position_sfen": record.initial_position_sfen,
        "moves": [shogi_move_record_to_json(move) for move in record.moves],
        "winner": record.winner,
        "end_reason": record.end_reason,
    }
    if record.metadata:
        payload["metadata"] = record.metadata
    return payload


def shogi_game_record_from_json(payload: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=shogi_actor_spec_from_json(_object_dict(payload.get("black_actor"))),
        white_actor=shogi_actor_spec_from_json(_object_dict(payload.get("white_actor"))),
        initial_position_sfen=str(payload.get("initial_position_sfen", shogi.Board().sfen())),
        moves=_move_records_from_payload(payload),
        winner=_normalize_winner(payload.get("winner")),
        end_reason=None if payload.get("end_reason") is None else str(payload["end_reason"]),
        metadata=_metadata_from_json(payload.get("metadata", {})),
    )


def shogi_move_records_from_usi_moves(moves: Sequence[str]) -> tuple[ShogiMoveRecord, ...]:
    return tuple(ShogiMoveRecord(action_usi=str(move)) for move in moves)


def shogi_game_record_from_usi_moves(
    moves: Sequence[str],
    *,
    black_actor: ShogiActorSpec,
    white_actor: ShogiActorSpec,
    winner: str | None = None,
    initial_position_sfen: str | None = None,
    end_reason: str | None = "game_over",
    metadata: dict[str, str | int | float | bool | None] | None = None,
) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_actor=black_actor,
        white_actor=white_actor,
        initial_position_sfen=initial_position_sfen or shogi.Board().sfen(),
        moves=shogi_move_records_from_usi_moves(moves),
        winner=_normalize_winner(winner),
        end_reason=end_reason,
        metadata=metadata or {},
    )


def shogi_move_record_to_json(record: ShogiMoveRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "action_usi": record.action_usi,
    }
    if record.decision_usi_info_lines:
        payload["decision_usi_info_lines"] = list(record.decision_usi_info_lines)
    if record.decision_telemetry is not None:
        payload["decision_telemetry"] = shogi_decision_telemetry_to_json(record.decision_telemetry)
    return payload


def shogi_move_record_from_json(payload: dict[str, object]) -> ShogiMoveRecord:
    info_lines = tuple(str(line) for line in _object_list(payload.get("decision_usi_info_lines", [])))
    telemetry = shogi_decision_telemetry_from_json(payload.get("decision_telemetry"))
    if telemetry is None:
        info_lines, telemetry = _migrate_legacy_performance_info_lines(info_lines)
    return ShogiMoveRecord(
        action_usi=str(payload["action_usi"]),
        decision_usi_info_lines=info_lines,
        decision_telemetry=telemetry,
    )


def shogi_decision_telemetry_to_json(telemetry: ShogiDecisionTelemetry) -> dict[str, object]:
    payload: dict[str, object] = {}
    if telemetry.move_performance is not None:
        payload["move_performance"] = telemetry.move_performance
    if telemetry.batch_performance is not None:
        payload["batch_performance"] = telemetry.batch_performance
    if telemetry.search_evidence is not None:
        payload["search_evidence"] = telemetry.search_evidence
    return payload


def shogi_decision_telemetry_from_json(value: object) -> ShogiDecisionTelemetry | None:
    if value is None:
        return None
    payload = _object_dict(value)
    return ShogiDecisionTelemetry(
        move_performance=_optional_object_dict(payload.get("move_performance")),
        batch_performance=_optional_object_dict(payload.get("batch_performance")),
        search_evidence=_optional_object_dict(payload.get("search_evidence")),
    )


def shogi_actor_spec_to_json(spec: ShogiActorSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def shogi_actor_spec_from_json(payload: dict[str, object]) -> ShogiActorSpec:
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("actor settings must be an object")
    return ShogiActorSpec(
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        settings={str(key): _json_scalar(value) for key, value in settings.items()},
    )


def shogi_winner_to_side_code(winner: str | None) -> str | None:
    if winner == "black":
        return "b"
    if winner == "white":
        return "w"
    return None


def shogi_side_code_to_winner(winner: str | None) -> str | None:
    if winner == "b":
        return "black"
    if winner == "w":
        return "white"
    if winner in {"black", "white"}:
        return winner
    return None


def _move_records_from_payload(payload: dict[str, object]) -> tuple[ShogiMoveRecord, ...]:
    if "moves" in payload:
        moves = _object_list(payload["moves"])
        if all(isinstance(move, str) for move in moves):
            return shogi_move_records_from_usi_moves(tuple(str(move) for move in moves))
        return tuple(shogi_move_record_from_json(_object_dict(move)) for move in moves)
    if "transitions" in payload:
        return tuple(
            shogi_move_record_from_json(_legacy_transition_to_move_json(_object_dict(transition)))
            for transition in _object_list(payload["transitions"])
        )
    raise ValueError("shogi game record must contain moves")


def _legacy_transition_to_move_json(payload: dict[str, object]) -> dict[str, object]:
    move_payload: dict[str, object] = {
        "action_usi": str(payload["action_usi"]),
    }
    if "decision_usi_info_lines" in payload:
        move_payload["decision_usi_info_lines"] = payload["decision_usi_info_lines"]
    if "decision_telemetry" in payload:
        move_payload["decision_telemetry"] = payload["decision_telemetry"]
    return move_payload


def _migrate_legacy_performance_info_lines(
    info_lines: tuple[str, ...],
) -> tuple[tuple[str, ...], ShogiDecisionTelemetry | None]:
    user_info_lines: list[str] = []
    move_performance: dict[str, object] | None = None
    batch_performance: dict[str, object] | None = None
    for line in info_lines:
        if line.startswith("info string intrep_performance "):
            move_performance = _parse_legacy_performance_payload(line, prefix="info string intrep_performance ")
            continue
        if line.startswith("info string intrep_batch_performance "):
            batch_performance = _parse_legacy_performance_payload(line, prefix="info string intrep_batch_performance ")
            continue
        user_info_lines.append(line)
    if move_performance is None and batch_performance is None:
        return tuple(user_info_lines), None
    return tuple(user_info_lines), ShogiDecisionTelemetry(
        move_performance=move_performance,
        batch_performance=batch_performance,
    )


def _parse_legacy_performance_payload(line: str, *, prefix: str) -> dict[str, object]:
    return _object_dict(json.loads(line[len(prefix) :]))


def _normalize_winner(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text in {"black", "white"}:
        return text
    if text in {"b", "w"}:
        return shogi_side_code_to_winner(text)
    raise ValueError("winner must be black, white, b, w, or null")


def _metadata_from_json(value: object) -> dict[str, str | int | float | bool | None]:
    payload = _object_dict(value)
    return {str(key): _json_scalar(item) for key, item in payload.items()}


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _optional_object_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    return _object_dict(value)


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("expected JSON scalar")
