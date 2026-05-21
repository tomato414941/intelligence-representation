from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import shogi


@dataclass(frozen=True)
class ShogiMoveChoiceExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    policy_targets: dict[str, float] | None = None
    policy_target_source: str = "chosen_move"
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
class ShogiMovePolicyValueExample:
    position_sfen: str
    legal_moves: tuple[str, ...]
    chosen_move: str
    policy_targets: dict[str, float] | None = None
    value_target: float | None = None
    policy_target_source: str = "chosen_move"
    value_target_source: str = "winner"
    game_index: int | None = None
    ply_index: int | None = None

    def __post_init__(self) -> None:
        _validate_policy_fields(self)
        _validate_value_target(self.value_target)


def load_shogi_move_policy_value_examples_jsonl(path: str | Path, *, max_examples: int | None = None) -> list[ShogiMovePolicyValueExample]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    examples: list[ShogiMovePolicyValueExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if max_examples is not None and len(examples) >= max_examples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(shogi_move_policy_value_example_from_json(json.loads(stripped)))
    return examples


def write_shogi_move_policy_value_examples_jsonl(path: str | Path, examples: Sequence[ShogiMovePolicyValueExample]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(shogi_move_policy_value_example_to_json(example), ensure_ascii=False) + "\n")


def shogi_move_policy_value_example_to_json(example: ShogiMovePolicyValueExample) -> dict[str, object]:
    payload: dict[str, object] = {
        "position_sfen": example.position_sfen,
        "legal_moves": list(example.legal_moves),
        "chosen_move": example.chosen_move,
    }
    if example.policy_targets is not None:
        payload["policy_targets"] = dict(sorted(example.policy_targets.items()))
    payload["policy_target_source"] = example.policy_target_source
    if example.value_target is not None:
        payload["value_target"] = example.value_target
    payload["value_target_source"] = example.value_target_source
    if example.game_index is not None:
        payload["game_index"] = example.game_index
    if example.ply_index is not None:
        payload["ply_index"] = example.ply_index
    return payload


def shogi_move_policy_value_example_from_json(payload: object) -> ShogiMovePolicyValueExample:
    if not isinstance(payload, dict):
        raise ValueError("shogi policy/value example must be an object")
    policy_targets = payload.get("policy_targets")
    if policy_targets is not None:
        if not isinstance(policy_targets, dict):
            raise ValueError("policy_targets must be an object")
        policy_targets = {str(move): float(weight) for move, weight in policy_targets.items()}
    return ShogiMovePolicyValueExample(
        position_sfen=str(payload["position_sfen"]),
        legal_moves=tuple(str(move) for move in _required_list(payload, "legal_moves")),
        chosen_move=str(payload["chosen_move"]),
        policy_targets=policy_targets,
        value_target=_optional_float(payload.get("value_target")),
        policy_target_source=str(payload.get("policy_target_source", "chosen_move")),
        value_target_source=str(payload.get("value_target_source", "winner")),
        game_index=_optional_int(payload.get("game_index")),
        ply_index=_optional_int(payload.get("ply_index")),
    )


def shogi_move_choice_example_from_board(board: shogi.Board, chosen_move: str) -> ShogiMoveChoiceExample:
    legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
    return ShogiMoveChoiceExample(
        position_sfen=board.sfen(),
        legal_moves=legal_moves,
        chosen_move=chosen_move,
    )


def _required_list(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _validate_policy_fields(example: ShogiMoveChoiceExample | ShogiMovePolicyValueExample) -> None:
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
