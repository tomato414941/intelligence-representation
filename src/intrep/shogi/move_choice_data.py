from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from intrep.shogi.game_record import load_shogi_game_records_jsonl, shogi_game_winner_to_legacy_side
from intrep.shogi.move_choice import (
    ShogiMoveChoiceExample,
    shogi_move_choice_examples_from_usi_moves,
    shogi_move_choice_examples_from_usi_moves_with_winner,
)


def load_shogi_move_choice_examples_from_game_records_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        winner = shogi_game_winner_to_legacy_side(record.winner)
        if winner is None:
            game_examples = shogi_move_choice_examples_from_usi_moves(record.moves)
        else:
            game_examples = shogi_move_choice_examples_from_usi_moves_with_winner(record.moves, winner=winner)
        examples.extend(_with_game_metadata(game_examples, game_index=game_index))
    return examples


def _with_game_metadata(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    game_index: int,
) -> list[ShogiMoveChoiceExample]:
    return [
        replace(example, game_index=game_index, ply_index=ply_index)
        for ply_index, example in enumerate(examples)
    ]
