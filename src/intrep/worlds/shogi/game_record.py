from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class PlayerSpec:
    kind: str
    name: str
    settings: dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class ShogiGamePlyRecord:
    side: str
    position: str
    bestmove: str
    ponder: str | None = None
    usi_info_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShogiGameRecord:
    black_player: PlayerSpec
    white_player: PlayerSpec
    plies: tuple[ShogiGamePlyRecord, ...]
    winner: str | None = None
    end_reason: str | None = None


def load_shogi_game_records_jsonl(path: str | Path) -> list[ShogiGameRecord]:
    records: list[ShogiGameRecord] = []
    for record in iter_shogi_game_records_jsonl(path):
        records.append(record)
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
            if record.plies:
                yield record


def write_shogi_game_records_jsonl(path: str | Path, records: Sequence[ShogiGameRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for record in records:
        if not record.plies:
            continue
        lines.append(json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True))
    if not lines:
        raise ValueError("records must contain at least one non-empty game")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def shogi_game_record_to_json(record: ShogiGameRecord) -> dict[str, object]:
    return {
        "black_player": player_spec_to_json(record.black_player),
        "white_player": player_spec_to_json(record.white_player),
        "plies": [shogi_game_ply_record_to_json(ply) for ply in record.plies],
        "winner": record.winner,
        "end_reason": record.end_reason,
    }


def shogi_game_record_from_json(payload: dict[str, object]) -> ShogiGameRecord:
    return ShogiGameRecord(
        black_player=player_spec_from_json(_object_dict(payload.get("black_player"))),
        white_player=player_spec_from_json(_object_dict(payload.get("white_player"))),
        plies=tuple(shogi_game_ply_record_from_json(_object_dict(ply)) for ply in _object_list(payload.get("plies"))),
        winner=_normalize_winner(payload.get("winner")),
        end_reason=None if payload.get("end_reason") is None else str(payload["end_reason"]),
    )


def shogi_game_ply_record_to_json(record: ShogiGamePlyRecord) -> dict[str, object]:
    return {
        "side": record.side,
        "position": record.position,
        "bestmove": record.bestmove,
        "ponder": record.ponder,
        "usi_info_lines": list(record.usi_info_lines),
    }


def shogi_game_ply_record_from_json(payload: dict[str, object]) -> ShogiGamePlyRecord:
    return ShogiGamePlyRecord(
        side=str(payload["side"]),
        position=str(payload["position"]),
        bestmove=str(payload["bestmove"]),
        ponder=None if payload.get("ponder") is None else str(payload["ponder"]),
        usi_info_lines=tuple(str(line) for line in _object_list(payload.get("usi_info_lines", []))),
    )


def player_spec_to_json(spec: PlayerSpec) -> dict[str, object]:
    return {
        "kind": spec.kind,
        "name": spec.name,
        "settings": spec.settings,
    }


def player_spec_from_json(payload: dict[str, object]) -> PlayerSpec:
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("player settings must be an object")
    return PlayerSpec(
        kind=str(payload["kind"]),
        name=str(payload["name"]),
        settings={
            str(key): _json_scalar(value)
            for key, value in settings.items()
        },
    )


def _normalize_winner(value: object) -> str | None:
    if value in {"black", "white"}:
        return str(value)
    if value == "b":
        return "black"
    if value == "w":
        return "white"
    if value is None:
        return None
    raise ValueError("winner must be black, white, b, w, or null")


def shogi_game_winner_to_legacy_side(winner: str | None) -> str | None:
    if winner == "black":
        return "b"
    if winner == "white":
        return "w"
    return None


def legacy_side_to_shogi_game_winner(winner: str | None) -> str | None:
    if winner == "b":
        return "black"
    if winner == "w":
        return "white"
    return None


def shogi_game_ply_records_from_usi_moves(moves: Sequence[str]) -> tuple[ShogiGamePlyRecord, ...]:
    previous_moves: list[str] = []
    records: list[ShogiGamePlyRecord] = []
    for ply_index, move in enumerate(moves):
        side = "black" if ply_index % 2 == 0 else "white"
        position = "position startpos" if not previous_moves else "position startpos moves " + " ".join(previous_moves)
        records.append(ShogiGamePlyRecord(side=side, position=position, bestmove=move))
        previous_moves.append(move)
    return tuple(records)


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError("player setting values must be JSON scalars")
