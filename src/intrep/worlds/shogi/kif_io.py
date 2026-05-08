from __future__ import annotations

from pathlib import Path
from typing import Sequence

import shogi.KIF

from intrep.worlds.shogi.game_record import (
    ShogiActorSpec,
    ShogiGameRecord,
    shogi_game_transitions_from_usi_moves,
    shogi_side_code_to_winner,
    write_shogi_game_records_jsonl,
)
from intrep.tasks.shogi_policy_value.data import shogi_move_choice_examples_from_game_record
from intrep.tasks.shogi_policy_value.examples import ShogiMoveChoiceExample


def load_kif_game(path: str | Path, *, encoding: str = "cp932") -> tuple[str, ...]:
    return load_kif_game_record(path, encoding=encoding)[0]


def load_kif_game_record(path: str | Path, *, encoding: str = "cp932") -> tuple[tuple[str, ...], str | None]:
    text = Path(path).read_text(encoding=encoding)
    parsed_games = shogi.KIF.Parser.parse_str(text)
    if not parsed_games:
        raise ValueError("KIF file must contain at least one game")
    game = parsed_games[0]
    winner = game.get("win")
    if winner not in {"b", "w"}:
        winner = None
    return tuple(game["moves"]), winner


def load_shogi_game_record_from_kif_file(path: str | Path) -> ShogiGameRecord:
    moves, winner = load_kif_game_record(path)
    normalized_winner = shogi_side_code_to_winner(winner)
    return ShogiGameRecord(
        black_actor=ShogiActorSpec(kind="kif", name="black", settings={}),
        white_actor=ShogiActorSpec(kind="kif", name="white", settings={}),
        initial_position_sfen=shogi.Board().sfen(),
        transitions=shogi_game_transitions_from_usi_moves(moves, winner=normalized_winner),
        winner=normalized_winner,
    )


def load_shogi_move_choice_examples_from_kif_file(path: str | Path) -> list[ShogiMoveChoiceExample]:
    return shogi_move_choice_examples_from_game_record(load_shogi_game_record_from_kif_file(path))


def convert_kif_files_to_game_records_jsonl(
    kif_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    max_games: int | None = None,
) -> int:
    records: list[ShogiGameRecord] = []
    for path in kif_paths:
        if max_games is not None and len(records) >= max_games:
            break
        records.append(load_shogi_game_record_from_kif_file(path))
    write_shogi_game_records_jsonl(output_path, records)
    return len(records)
