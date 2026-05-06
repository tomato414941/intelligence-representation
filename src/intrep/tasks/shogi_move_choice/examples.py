from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import shogi

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised in lightweight preprocessing environments.
    torch = None
    TorchDataset = object

@dataclass(frozen=True)
class ShogiMoveChoiceExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    policy_targets: dict[str, float] | None = None
    value_target: float | None = None
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        if not self.legal_moves:
            raise ValueError("legal_moves must not be empty")
        if self.chosen_move not in self.legal_moves:
            raise ValueError("chosen_move must be included in legal_moves")
        if self.policy_targets is not None:
            if not self.policy_targets:
                raise ValueError("policy_targets must not be empty")
            unknown_moves = set(self.policy_targets) - set(self.legal_moves)
            if unknown_moves:
                raise ValueError("policy_targets moves must be included in legal_moves")
            if any(weight < 0.0 for weight in self.policy_targets.values()):
                raise ValueError("policy_targets weights must be non-negative")
            if sum(self.policy_targets.values()) <= 0.0:
                raise ValueError("policy_targets weights must have positive sum")
        if self.value_target is not None and not -1.0 <= self.value_target <= 1.0:
            raise ValueError("value_target must be between -1.0 and 1.0")
        if self.game_index is not None and self.game_index < 0:
            raise ValueError("game_index must be non-negative")
        if self.ply_index is not None and self.ply_index < 0:
            raise ValueError("ply_index must be non-negative")


def shogi_move_choice_example_from_board(board: shogi.Board, chosen_move: str) -> ShogiMoveChoiceExample:
    legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
    return ShogiMoveChoiceExample(
        position_sfen=board.sfen(),
        legal_moves=legal_moves,
        chosen_move=chosen_move,
    )


class ShogiMoveChoiceDataset(TorchDataset):
    def __init__(self, examples: Sequence[ShogiMoveChoiceExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.max_choice_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiMoveChoiceDataset items")
        from intrep.worlds.shogi.move_encoding import shogi_candidate_move_features
        from intrep.worlds.shogi.position_encoding import shogi_position_token_ids_from_sfen

        example = self.examples[index]
        position_token_ids = shogi_position_token_ids_from_sfen(example.position_sfen)
        candidate_move_features = shogi_candidate_move_features(
            example.legal_moves,
            max_choice_count=self.max_choice_count,
        )
        move_index = example.legal_moves.index(example.chosen_move)
        candidate_mask = torch.zeros(self.max_choice_count, dtype=torch.bool)
        candidate_mask[: len(example.legal_moves)] = True
        policy_targets = torch.zeros(self.max_choice_count, dtype=torch.float32)
        if example.policy_targets is None:
            policy_targets[move_index] = 1.0
        else:
            total = sum(example.policy_targets.values())
            for move, weight in example.policy_targets.items():
                policy_targets[example.legal_moves.index(move)] = float(weight) / total
        value_target = float("nan") if example.value_target is None else example.value_target
        return (
            position_token_ids,
            candidate_move_features,
            candidate_mask,
            torch.tensor(move_index, dtype=torch.long),
            policy_targets,
            torch.tensor(value_target, dtype=torch.float32),
        )

