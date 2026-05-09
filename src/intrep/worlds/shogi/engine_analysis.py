from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_actor_spec_from_json,
    shogi_actor_spec_to_json,
)


@dataclass(frozen=True)
class ShogiAnalysisPosition:
    position_sfen: str
    legal_moves: tuple[str, ...]


@dataclass(frozen=True)
class ShogiEngineAnalysis:
    # This is intentionally narrow: analysis of a shogi position by a shogi
    # engine. General annotations, search traces, or cross-world evidence may
    # deserve a broader abstraction later, but this type should not claim that
    # scope before it is needed.
    position_sfen: str
    legal_moves: tuple[str, ...]
    engine: ShogiActorSpec
    usi_info_lines: tuple[str, ...]
    created_at: str | None = None


class ShogiUsiGoResult(Protocol):
    info_lines: tuple[str, ...]


class ShogiUsiSession(Protocol):
    def position(self, command: str) -> None:
        ...

    def go(self) -> ShogiUsiGoResult:
        ...


def shogi_analysis_positions_from_game_records(records: Sequence[ShogiGameRecord]) -> list[ShogiAnalysisPosition]:
    positions: list[ShogiAnalysisPosition] = []
    seen_position_sfen: set[str] = set()
    for record in records:
        for transition in record.transitions:
            if transition.position_sfen in seen_position_sfen:
                continue
            seen_position_sfen.add(transition.position_sfen)
            positions.append(
                ShogiAnalysisPosition(
                    position_sfen=transition.position_sfen,
                    legal_moves=transition.legal_moves,
                )
            )
    return positions


def analyze_shogi_position_with_usi_session(
    position: ShogiAnalysisPosition,
    *,
    engine: ShogiActorSpec,
    session: ShogiUsiSession,
    created_at: str | None = None,
) -> ShogiEngineAnalysis:
    session.position(f"position sfen {position.position_sfen}")
    result = session.go()
    return ShogiEngineAnalysis(
        position_sfen=position.position_sfen,
        legal_moves=position.legal_moves,
        engine=engine,
        usi_info_lines=result.info_lines,
        created_at=created_at,
    )


def write_shogi_engine_analysis_jsonl(path: str | Path, records: Sequence[ShogiEngineAnalysis]) -> None:
    if not records:
        raise ValueError("records must contain at least one analysis")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(shogi_engine_analysis_to_json(record), separators=(",", ":"), sort_keys=True)
        for record in records
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_shogi_engine_analysis_jsonl(path: str | Path) -> list[ShogiEngineAnalysis]:
    records = list(iter_shogi_engine_analysis_jsonl(path))
    if not records:
        raise ValueError("shogi engine analysis jsonl must contain at least one analysis")
    return records


def iter_shogi_engine_analysis_jsonl(path: str | Path) -> Iterator[ShogiEngineAnalysis]:
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                yield shogi_engine_analysis_from_json(json.loads(stripped))


def shogi_engine_analysis_to_json(record: ShogiEngineAnalysis) -> dict[str, object]:
    return {
        "position_sfen": record.position_sfen,
        "legal_moves": list(record.legal_moves),
        "engine": shogi_actor_spec_to_json(record.engine),
        "usi_info_lines": list(record.usi_info_lines),
        "created_at": record.created_at,
    }


def shogi_engine_analysis_from_json(payload: dict[str, object]) -> ShogiEngineAnalysis:
    return ShogiEngineAnalysis(
        position_sfen=str(payload["position_sfen"]),
        legal_moves=tuple(str(move) for move in _object_list(payload.get("legal_moves"))),
        engine=shogi_actor_spec_from_json(_object_dict(payload.get("engine"))),
        usi_info_lines=tuple(str(line) for line in _object_list(payload.get("usi_info_lines", []))),
        created_at=None if payload.get("created_at") is None else str(payload["created_at"]),
    )


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value
