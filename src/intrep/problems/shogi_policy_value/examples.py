from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class TensorizedShogiPolicyValueSample:
    position_token_ids: torch.Tensor
    candidate_move_features: torch.Tensor
    label: torch.Tensor
    policy_targets: torch.Tensor
    value_target: torch.Tensor


ShogiPolicyValueDatasetItem = ShogiPolicyValueExample | TensorizedShogiPolicyValueSample


def load_shogi_policy_value_examples_jsonl(path: str | Path, *, max_examples: int | None = None) -> list[ShogiPolicyValueExample]:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    examples: list[ShogiPolicyValueExample] = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if max_examples is not None and len(examples) >= max_examples:
                break
            stripped = line.strip()
            if not stripped:
                continue
            examples.append(shogi_policy_value_example_from_json(json.loads(stripped)))
    return examples


def write_shogi_policy_value_examples_jsonl(path: str | Path, examples: Sequence[ShogiPolicyValueExample]) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(shogi_policy_value_example_to_json(example), ensure_ascii=False) + "\n")


def shogi_policy_value_example_to_json(example: ShogiPolicyValueExample) -> dict[str, object]:
    payload: dict[str, object] = {
        "position_sfen": example.position_sfen,
        "legal_moves": list(example.legal_moves),
        "chosen_move": example.chosen_move,
    }
    if example.policy_targets is not None:
        payload["policy_targets"] = dict(sorted(example.policy_targets.items()))
    if example.value_target is not None:
        payload["value_target"] = example.value_target
    if example.game_index is not None:
        payload["game_index"] = example.game_index
    if example.ply_index is not None:
        payload["ply_index"] = example.ply_index
    return payload


def shogi_policy_value_example_from_json(payload: object) -> ShogiPolicyValueExample:
    if not isinstance(payload, dict):
        raise ValueError("shogi policy/value example must be an object")
    policy_targets = payload.get("policy_targets")
    if policy_targets is not None:
        if not isinstance(policy_targets, dict):
            raise ValueError("policy_targets must be an object")
        policy_targets = {str(move): float(weight) for move, weight in policy_targets.items()}
    return ShogiPolicyValueExample(
        position_sfen=str(payload["position_sfen"]),
        legal_moves=tuple(str(move) for move in _required_list(payload, "legal_moves")),
        chosen_move=str(payload["chosen_move"]),
        policy_targets=policy_targets,
        value_target=_optional_float(payload.get("value_target")),
        game_index=_optional_int(payload.get("game_index")),
        ply_index=_optional_int(payload.get("ply_index")),
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
    def __init__(self, examples: Sequence[ShogiPolicyValueDatasetItem]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = examples
        self.max_choice_count = int(
            getattr(examples, "max_choice_count", None) or max(_choice_count(example) for example in self.examples)
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiPolicyValueDataset items")
        sample = _tensorized_policy_value_sample(self.examples[index])
        return (
            sample.position_token_ids,
            _pad_candidate_move_features(sample.candidate_move_features, max_choice_count=self.max_choice_count),
            _candidate_mask(_choice_count(sample), max_choice_count=self.max_choice_count),
            sample.label,
            _pad_policy_targets(sample.policy_targets, max_choice_count=self.max_choice_count),
            sample.value_target,
        )


def tensorize_shogi_policy_value_example(example: ShogiPolicyValueExample) -> TensorizedShogiPolicyValueSample:
    position_token_ids, candidate_move_features, _candidate_mask, move_index, policy_targets = _policy_sample(
        example,
        max_choice_count=len(example.legal_moves),
    )
    return TensorizedShogiPolicyValueSample(
        position_token_ids=position_token_ids,
        candidate_move_features=candidate_move_features,
        label=move_index,
        policy_targets=policy_targets,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_shogi_policy_value_examples(
    examples: Sequence[ShogiPolicyValueExample],
) -> list[TensorizedShogiPolicyValueSample]:
    return [tensorize_shogi_policy_value_example(example) for example in examples]


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


def _tensorized_policy_value_sample(example: ShogiPolicyValueDatasetItem) -> TensorizedShogiPolicyValueSample:
    if isinstance(example, TensorizedShogiPolicyValueSample):
        return example
    return tensorize_shogi_policy_value_example(example)


def _choice_count(example: ShogiPolicyValueDatasetItem) -> int:
    if isinstance(example, TensorizedShogiPolicyValueSample):
        return int(example.candidate_move_features.shape[0])
    return len(example.legal_moves)


def _candidate_mask(choice_count: int, *, max_choice_count: int) -> torch.Tensor:
    candidate_mask = torch.zeros(max_choice_count, dtype=torch.bool)
    candidate_mask[:choice_count] = True
    return candidate_mask


def _pad_candidate_move_features(candidate_move_features: torch.Tensor, *, max_choice_count: int) -> torch.Tensor:
    if candidate_move_features.shape[0] == max_choice_count:
        return candidate_move_features
    padded = torch.zeros(
        (max_choice_count, candidate_move_features.shape[1]),
        dtype=candidate_move_features.dtype,
    )
    padded[: candidate_move_features.shape[0]] = candidate_move_features
    return padded


def _pad_policy_targets(policy_targets: torch.Tensor, *, max_choice_count: int) -> torch.Tensor:
    if policy_targets.shape[0] == max_choice_count:
        return policy_targets
    padded = torch.zeros(max_choice_count, dtype=policy_targets.dtype)
    padded[: policy_targets.shape[0]] = policy_targets
    return padded


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
