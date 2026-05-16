from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import shogi

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_record_from_usi_moves,
    shogi_side_code_to_winner,
)


@dataclass(frozen=True)
class ShogiMoveListRecord:
    moves: tuple[str, ...]
    winner: str | None


def iter_shogi_move_list_records_jsonl(
    path: str | Path,
    *,
    start_index: int = 0,
    end_index: int | None = None,
) -> Iterator[tuple[int, ShogiMoveListRecord]]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be greater than or equal to start_index")

    with Path(path).open(encoding="utf-8") as file:
        for index, line in enumerate(file):
            if index < start_index:
                continue
            if end_index is not None and index >= end_index:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield index, shogi_move_list_record_from_json(json.loads(stripped))


def shogi_move_list_record_from_json(payload: dict[str, object]) -> ShogiMoveListRecord:
    moves = payload.get("moves")
    if not isinstance(moves, list):
        raise ValueError("shogi move-list record must contain moves")
    return ShogiMoveListRecord(
        moves=tuple(str(move) for move in moves),
        winner=shogi_side_code_to_winner(None if payload.get("winner") is None else str(payload["winner"])),
    )


def shogi_game_record_from_move_list_record(
    record: ShogiMoveListRecord,
    *,
    source_name: str,
    source_record_index: int,
    end_reason: str = "game_over",
) -> ShogiGameRecord:
    settings: dict[str, str | int | float | bool | None] = {
        "source": source_name,
        "source_record_index": source_record_index,
    }
    return shogi_game_record_from_usi_moves(
        record.moves,
        black_actor=ShogiActorSpec(kind="recorded", name="black", settings=settings),
        white_actor=ShogiActorSpec(kind="recorded", name="white", settings=settings),
        initial_position_sfen=shogi.Board().sfen(),
        winner=record.winner,
        end_reason=end_reason,
    )
