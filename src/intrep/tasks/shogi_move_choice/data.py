from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from intrep.worlds.shogi.game_record import load_shogi_game_records_jsonl, shogi_game_winner_to_legacy_side
from intrep.tasks.shogi_move_choice.examples import (
    ShogiMoveChoiceExample,
    shogi_move_choice_examples_from_usi_moves,
    shogi_move_choice_examples_from_usi_moves_with_winner,
)


def load_shogi_move_choice_examples_from_game_records_jsonl(path: str | Path) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for game_index, record in enumerate(load_shogi_game_records_jsonl(path)):
        winner = shogi_game_winner_to_legacy_side(record.winner)
        moves = tuple(ply.bestmove for ply in record.plies)
        if winner is None:
            game_examples = shogi_move_choice_examples_from_usi_moves(moves)
        else:
            game_examples = shogi_move_choice_examples_from_usi_moves_with_winner(moves, winner=winner)
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
