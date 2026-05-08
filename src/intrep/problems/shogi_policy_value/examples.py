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
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        _validate_policy_fields(self)


@dataclass(frozen=True)
class ShogiPositionValueExample:
    position_sfen: str
    value_target: float | None = None
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        if not self.position_sfen:
            raise ValueError("position_sfen must not be empty")
        _validate_value_target(self.value_target)
        _validate_metadata(self.game_index, self.ply_index)


@dataclass(frozen=True)
class ShogiPolicyValueExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    policy_targets: dict[str, float] | None = None
    value_target: float | None = None
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        _validate_policy_fields(self)
        _validate_value_target(self.value_target)


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
        return _policy_sample(
            self.examples[index],
            max_choice_count=self.max_choice_count,
        )


class ShogiPolicyValueDataset(TorchDataset):
    def __init__(self, examples: Sequence[ShogiPolicyValueExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.max_choice_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiPolicyValueDataset items")
        position_token_ids, candidate_move_features, candidate_mask, move_index, policy_targets = _policy_sample(
            self.examples[index],
            max_choice_count=self.max_choice_count,
        )
        value_target = self.examples[index].value_target
        return (
            position_token_ids,
            candidate_move_features,
            candidate_mask,
            move_index,
            policy_targets,
            torch.tensor(float("nan") if value_target is None else value_target, dtype=torch.float32),
        )


def _policy_sample(
    example: ShogiMoveChoiceExample | ShogiPolicyValueExample,
    *,
    max_choice_count: int,
):
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi policy samples")
    from intrep.worlds.shogi.move_encoding import shogi_candidate_move_features
    from intrep.worlds.shogi.position_encoding import shogi_position_token_ids_from_sfen

    position_token_ids = shogi_position_token_ids_from_sfen(example.position_sfen)
    candidate_move_features = shogi_candidate_move_features(
        example.legal_moves,
        max_choice_count=max_choice_count,
    )
    move_index = example.legal_moves.index(example.chosen_move)
    candidate_mask = torch.zeros(max_choice_count, dtype=torch.bool)
    candidate_mask[: len(example.legal_moves)] = True
    policy_targets = torch.zeros(max_choice_count, dtype=torch.float32)
    if example.policy_targets is None:
        policy_targets[move_index] = 1.0
    else:
        total = sum(example.policy_targets.values())
        for move, weight in example.policy_targets.items():
            policy_targets[example.legal_moves.index(move)] = float(weight) / total
    return (
        position_token_ids,
        candidate_move_features,
        candidate_mask,
        torch.tensor(move_index, dtype=torch.long),
        policy_targets,
    )


def _validate_policy_fields(example: ShogiMoveChoiceExample | ShogiPolicyValueExample) -> None:
    if not example.position_sfen:
        raise ValueError("position_sfen must not be empty")
    if not example.legal_moves:
        raise ValueError("legal_moves must not be empty")
    if example.chosen_move not in example.legal_moves:
        raise ValueError("chosen_move must be included in legal_moves")
    if example.policy_targets is not None:
        if not example.policy_targets:
            raise ValueError("policy_targets must not be empty")
        unknown_moves = set(example.policy_targets) - set(example.legal_moves)
        if unknown_moves:
            raise ValueError("policy_targets moves must be included in legal_moves")
        if any(weight < 0.0 for weight in example.policy_targets.values()):
            raise ValueError("policy_targets weights must be non-negative")
        if sum(example.policy_targets.values()) <= 0.0:
            raise ValueError("policy_targets weights must have positive sum")
    _validate_metadata(example.game_index, example.ply_index)


def _validate_value_target(value_target: float | None) -> None:
    if value_target is not None and not -1.0 <= value_target <= 1.0:
        raise ValueError("value_target must be between -1.0 and 1.0")


def _validate_metadata(game_index: int | None, ply_index: int | None) -> None:
    if game_index is not None and game_index < 0:
        raise ValueError("game_index must be non-negative")
    if ply_index is not None and ply_index < 0:
        raise ValueError("ply_index must be non-negative")
