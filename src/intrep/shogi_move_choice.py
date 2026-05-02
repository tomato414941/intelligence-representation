from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import shogi
import torch
from torch.utils.data import Dataset

from intrep.shogi_move_encoding import shogi_candidate_move_features
from intrep.shogi_position_encoding import shogi_position_token_ids_from_sfen


@dataclass(frozen=True)
class ShogiMoveChoiceExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    value_target: float | None = None

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        if not self.legal_moves:
            raise ValueError("legal_moves must not be empty")
        if self.chosen_move not in self.legal_moves:
            raise ValueError("chosen_move must be included in legal_moves")
        if self.value_target is not None and not -1.0 <= self.value_target <= 1.0:
            raise ValueError("value_target must be between -1.0 and 1.0")


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


def shogi_move_choice_examples_from_usi_moves_with_winner(
    moves: Sequence[str],
    *,
    winner: str,
) -> list[ShogiMoveChoiceExample]:
    if winner not in {"b", "w"}:
        raise ValueError("winner must be 'b' or 'w'")
    board = shogi.Board()
    examples: list[ShogiMoveChoiceExample] = []
    for move in moves:
        side_to_move = "b" if board.turn == shogi.BLACK else "w"
        value_target = 1.0 if side_to_move == winner else -1.0
        base_example = shogi_move_choice_example_from_board(board, move)
        examples.append(
            ShogiMoveChoiceExample(
                position_sfen=base_example.position_sfen,
                legal_moves=base_example.legal_moves,
                chosen_move=base_example.chosen_move,
                value_target=value_target,
            )
        )
        board.push_usi(move)
    return examples


class ShogiMoveChoiceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: Sequence[ShogiMoveChoiceExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.max_choice_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        position_token_ids = shogi_position_token_ids_from_sfen(example.position_sfen)
        candidate_move_features = shogi_candidate_move_features(
            example.legal_moves,
            max_choice_count=self.max_choice_count,
        )
        move_index = example.legal_moves.index(example.chosen_move)
        candidate_mask = torch.zeros(self.max_choice_count, dtype=torch.bool)
        candidate_mask[: len(example.legal_moves)] = True
        value_target = float("nan") if example.value_target is None else example.value_target
        return (
            position_token_ids,
            candidate_move_features,
            candidate_mask,
            torch.tensor(move_index, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32),
        )
