from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import shogi
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShogiMoveChoiceExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        if not self.legal_moves:
            raise ValueError("legal_moves must not be empty")
        if self.chosen_move not in self.legal_moves:
            raise ValueError("chosen_move must be included in legal_moves")


def shogi_move_choice_example_from_board(board: shogi.Board, chosen_move: str) -> ShogiMoveChoiceExample:
    legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
    return ShogiMoveChoiceExample(
        position_sfen=board.sfen(),
        legal_moves=legal_moves,
        chosen_move=chosen_move,
    )


def shogi_move_choice_examples_from_usi_moves(moves: Sequence[str]) -> list[ShogiMoveChoiceExample]:
    board = shogi.Board()
    examples: list[ShogiMoveChoiceExample] = []
    for move in moves:
        examples.append(shogi_move_choice_example_from_board(board, move))
        board.push_usi(move)
    return examples


class ShogiMoveChoiceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: Sequence[ShogiMoveChoiceExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.max_choice_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        move_index = example.legal_moves.index(example.chosen_move)
        candidate_mask = torch.zeros(self.max_choice_count, dtype=torch.bool)
        candidate_mask[: len(example.legal_moves)] = True
        return candidate_mask, torch.tensor(move_index, dtype=torch.long)
