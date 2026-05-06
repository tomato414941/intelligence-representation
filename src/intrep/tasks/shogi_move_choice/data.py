from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from intrep.tasks.shogi_move_choice.examples import (
    ShogiMoveChoiceExample,
)
from intrep.worlds.shogi.game_record import ShogiGameRecord, load_shogi_game_records_jsonl


def load_shogi_move_choice_examples_from_game_records_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        game_examples = shogi_move_choice_examples_from_game_record(record)
        examples.extend(_with_game_metadata(game_examples, game_index=game_index))
    return examples


def shogi_move_choice_examples_from_game_record(record: ShogiGameRecord) -> list[ShogiMoveChoiceExample]:
    return [
        ShogiMoveChoiceExample(
            position_sfen=transition.position_sfen,
            legal_moves=transition.legal_moves,
            chosen_move=transition.action_usi,
            value_target=_value_target(record.winner, transition.side),
        )
        for transition in record.transitions
    ]


def _with_game_metadata(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    game_index: int,
) -> list[ShogiMoveChoiceExample]:
    return [
        replace(example, game_index=game_index, ply_index=ply_index)
        for ply_index, example in enumerate(examples)
    ]


def _value_target(winner: str | None, side: str) -> float | None:
    if winner is None:
        return None
    return 1.0 if side == winner else -1.0
