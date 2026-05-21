from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import shogi
from intrep.representation.inputs.shogi_position_features.position_encoding import ShogiPositionFeatures, stack_shogi_position_features


ShogiPositionFeatureBuilder = Callable[[str], ShogiPositionFeatures]

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


@dataclass(frozen=True)
class LegalMovePolicyValueTensorSample:
    position_features: ShogiPositionFeatures
    legal_move_features: torch.Tensor
    label: torch.Tensor
    policy_targets: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class PolicyPlaneValueTensorSample:
    position_features: ShogiPositionFeatures
    policy_plane_targets: torch.Tensor
    policy_plane_legal_mask: torch.Tensor
    policy_plane_label: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class CompactPolicyPlaneValueTensorSample:
    position_features: ShogiPositionFeatures
    legal_action_indices: torch.Tensor
    target_action_indices: torch.Tensor
    target_weights: torch.Tensor
    policy_plane_label: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class LegalMovePolicyValueBatch:
    position_features: ShogiPositionFeatures
    legal_move_features: torch.Tensor
    legal_move_mask: torch.Tensor
    labels: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor

    def to(self, device: torch.device) -> "LegalMovePolicyValueBatch":
        return LegalMovePolicyValueBatch(
            position_features=self.position_features.to(device),
            legal_move_features=self.legal_move_features.to(device),
            legal_move_mask=self.legal_move_mask.to(device),
            labels=self.labels.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
        )


@dataclass(frozen=True)
class PolicyPlaneValueBatch:
    position_features: ShogiPositionFeatures
    legal_action_mask: torch.Tensor
    labels: torch.Tensor
    target_action_indices: torch.Tensor
    target_weights: torch.Tensor
    value_targets: torch.Tensor

    def to(self, device: torch.device) -> "PolicyPlaneValueBatch":
        return PolicyPlaneValueBatch(
            position_features=self.position_features.to(device),
            legal_action_mask=self.legal_action_mask.to(device),
            labels=self.labels.to(device),
            target_action_indices=self.target_action_indices.to(device),
            target_weights=self.target_weights.to(device),
            value_targets=self.value_targets.to(device),
        )


ShogiPolicyValueDatasetItem = (
    ShogiMovePolicyValueExample
    | LegalMovePolicyValueTensorSample
    | PolicyPlaneValueTensorSample
    | CompactPolicyPlaneValueTensorSample
)


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
    def __init__(
        self,
        examples: Sequence[ShogiMoveChoiceExample],
        *,
        position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
    ) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = tuple(examples)
        self.position_features_from_sfen = position_features_from_sfen
        self.max_legal_move_count = max(len(example.legal_moves) for example in self.examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiMoveChoiceDataset items")
        return _policy_sample(
            self.examples[index],
            max_legal_move_count=self.max_legal_move_count,
            position_features_from_sfen=self.position_features_from_sfen,
        )


class ShogiLegalMovePolicyValueDataset(TorchDataset):
    def __init__(
        self,
        examples: Sequence[ShogiPolicyValueDatasetItem],
        *,
        position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
    ) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = examples
        self.position_features_from_sfen = position_features_from_sfen
        self.max_legal_move_count = int(
            getattr(examples, "max_legal_move_count", None) or max(_choice_count(example) for example in self.examples)
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiLegalMovePolicyValueDataset items")
        return _legal_move_policy_value_tensor_sample(
            self.examples[index],
            position_features_from_sfen=self.position_features_from_sfen,
        )


def tensorize_legal_move_policy_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> LegalMovePolicyValueTensorSample:
    position_features, legal_move_features, _legal_move_mask, move_index, policy_targets = _policy_sample(
        example,
        max_legal_move_count=len(example.legal_moves),
        position_features_from_sfen=position_features_from_sfen,
    )
    return LegalMovePolicyValueTensorSample(
        position_features=position_features,
        legal_move_features=legal_move_features,
        label=move_index,
        policy_targets=policy_targets,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_legal_move_policy_value_examples(
    examples: Sequence[ShogiMovePolicyValueExample],
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> list[LegalMovePolicyValueTensorSample]:
    return [
        tensorize_legal_move_policy_value_example(
            example,
            position_features_from_sfen=position_features_from_sfen,
        )
        for example in examples
    ]


class ShogiPolicyPlaneValueDataset(TorchDataset):
    def __init__(
        self,
        examples: Sequence[ShogiPolicyValueDatasetItem],
        *,
        position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
    ) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        if any(isinstance(example, LegalMovePolicyValueTensorSample) for example in examples):
            raise ValueError("legal-move tensor samples cannot be used with ShogiPolicyPlaneValueDataset")
        self.examples = examples
        self.position_features_from_sfen = position_features_from_sfen

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiPolicyPlaneValueDataset items")
        return _compact_policy_plane_value_tensor_sample(
            self.examples[index],
            position_features_from_sfen=self.position_features_from_sfen,
        )


def collate_legal_move_policy_value_samples(
    samples: Sequence[LegalMovePolicyValueTensorSample],
) -> LegalMovePolicyValueBatch:
    if torch is None:
        raise RuntimeError("torch is required to collate shogi policy/value samples")
    max_legal_move_count = max(int(sample.legal_move_features.shape[0]) for sample in samples)
    return LegalMovePolicyValueBatch(
        position_features=stack_shogi_position_features([sample.position_features for sample in samples]),
        legal_move_features=torch.stack(
            [
                _pad_legal_move_features(sample.legal_move_features, max_legal_move_count=max_legal_move_count)
                for sample in samples
            ]
        ),
        legal_move_mask=torch.stack(
            [_legal_move_mask(_choice_count(sample), max_legal_move_count=max_legal_move_count) for sample in samples]
        ),
        labels=torch.stack([sample.label for sample in samples]),
        policy_targets=torch.stack(
            [_pad_policy_targets(sample.policy_targets, max_legal_move_count=max_legal_move_count) for sample in samples]
        ),
        value_targets=torch.stack([sample.value_target for sample in samples]),
    )


def collate_policy_plane_value_samples(
    samples: Sequence[CompactPolicyPlaneValueTensorSample],
) -> PolicyPlaneValueBatch:
    if torch is None:
        raise RuntimeError("torch is required to collate shogi policy-plane samples")
    from intrep.representation.outputs.shogi_policy_plane_encoding import SHOGI_POLICY_PLANE_ACTION_COUNT

    max_target_count = max(int(sample.target_action_indices.shape[0]) for sample in samples)
    legal_action_mask = torch.zeros((len(samples), SHOGI_POLICY_PLANE_ACTION_COUNT), dtype=torch.bool)
    for row_index, sample in enumerate(samples):
        legal_action_mask[row_index, sample.legal_action_indices.long()] = True
    return PolicyPlaneValueBatch(
        position_features=stack_shogi_position_features([sample.position_features for sample in samples]),
        legal_action_mask=legal_action_mask,
        labels=torch.stack([sample.policy_plane_label for sample in samples]),
        target_action_indices=torch.stack(
            [
                _pad_action_indices(sample.target_action_indices, max_action_count=max_target_count)
                for sample in samples
            ]
        ),
        target_weights=torch.stack(
            [_pad_target_weights(sample.target_weights, max_action_count=max_target_count) for sample in samples]
        ),
        value_targets=torch.stack([sample.value_target for sample in samples]),
    )


def tensorize_policy_plane_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> PolicyPlaneValueTensorSample:
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi policy-plane samples")
    from intrep.representation.outputs.shogi_policy_plane_encoding import (
        SHOGI_POLICY_PLANE_ACTION_COUNT,
        shogi_policy_plane_action_index,
        shogi_policy_plane_legal_mask,
    )
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    policy_plane_label = torch.tensor(
        shogi_policy_plane_action_index(example.chosen_move, turn=board.turn),
        dtype=torch.long,
    )
    policy_plane_targets = torch.zeros(SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.float32)
    if example.policy_targets is None:
        if example.policy_target_source != "chosen_move":
            raise ValueError(f"missing policy_targets for policy_target_source={example.policy_target_source}")
        policy_plane_targets[int(policy_plane_label.item())] = 1.0
    else:
        total = sum(example.policy_targets.values())
        for move, weight in example.policy_targets.items():
            action_index = shogi_policy_plane_action_index(move, turn=board.turn)
            policy_plane_targets[action_index] = float(weight) / total
    return PolicyPlaneValueTensorSample(
        position_features=position_features,
        policy_plane_targets=policy_plane_targets,
        policy_plane_legal_mask=shogi_policy_plane_legal_mask(board),
        policy_plane_label=policy_plane_label,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_policy_plane_value_examples(
    examples: Sequence[ShogiMovePolicyValueExample],
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> list[PolicyPlaneValueTensorSample]:
    return [
        tensorize_policy_plane_value_example(
            example,
            position_features_from_sfen=position_features_from_sfen,
        )
        for example in examples
    ]


def tensorize_compact_policy_plane_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> CompactPolicyPlaneValueTensorSample:
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi policy-plane samples")
    from intrep.representation.outputs.shogi_policy_plane_encoding import shogi_policy_plane_action_index
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    legal_action_indices = torch.tensor(
        [shogi_policy_plane_action_index(move, turn=board.turn) for move in example.legal_moves],
        dtype=torch.long,
    )
    policy_plane_label = torch.tensor(
        shogi_policy_plane_action_index(example.chosen_move, turn=board.turn),
        dtype=torch.long,
    )
    if example.policy_targets is None:
        if example.policy_target_source != "chosen_move":
            raise ValueError(f"missing policy_targets for policy_target_source={example.policy_target_source}")
        target_action_indices = policy_plane_label.reshape(1)
        target_weights = torch.ones(1, dtype=torch.float32)
    else:
        total = sum(example.policy_targets.values())
        target_action_indices = torch.tensor(
            [shogi_policy_plane_action_index(move, turn=board.turn) for move in example.policy_targets],
            dtype=torch.long,
        )
        target_weights = torch.tensor(
            [float(weight) / total for weight in example.policy_targets.values()],
            dtype=torch.float32,
        )
    return CompactPolicyPlaneValueTensorSample(
        position_features=position_features,
        legal_action_indices=legal_action_indices,
        target_action_indices=target_action_indices,
        target_weights=target_weights,
        policy_plane_label=policy_plane_label,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_compact_policy_plane_value_examples(
    examples: Sequence[ShogiMovePolicyValueExample],
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> list[CompactPolicyPlaneValueTensorSample]:
    return [
        tensorize_compact_policy_plane_value_example(
            example,
            position_features_from_sfen=position_features_from_sfen,
        )
        for example in examples
    ]


def _policy_sample(
    example: ShogiMoveChoiceExample | ShogiMovePolicyValueExample,
    *,
    max_legal_move_count: int,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
):
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi policy samples")
    from intrep.representation.outputs.shogi_legal_move_encoding import shogi_legal_move_features
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    legal_move_features = shogi_legal_move_features(
        example.legal_moves,
        turn=board.turn,
        max_legal_move_count=max_legal_move_count,
    )
    move_index = example.legal_moves.index(example.chosen_move)
    legal_move_mask = torch.zeros(max_legal_move_count, dtype=torch.bool)
    legal_move_mask[: len(example.legal_moves)] = True
    policy_targets = torch.zeros(max_legal_move_count, dtype=torch.float32)
    if example.policy_targets is None:
        if example.policy_target_source != "chosen_move":
            raise ValueError(f"missing policy_targets for policy_target_source={example.policy_target_source}")
        policy_targets[move_index] = 1.0
    else:
        total = sum(example.policy_targets.values())
        for move, weight in example.policy_targets.items():
            policy_targets[example.legal_moves.index(move)] = float(weight) / total
    return (
        position_features,
        legal_move_features,
        legal_move_mask,
        torch.tensor(move_index, dtype=torch.long),
        policy_targets,
    )


def _legal_move_policy_value_tensor_sample(
    example: ShogiPolicyValueDatasetItem,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> LegalMovePolicyValueTensorSample:
    if isinstance(example, LegalMovePolicyValueTensorSample):
        return example
    if isinstance(example, (PolicyPlaneValueTensorSample, CompactPolicyPlaneValueTensorSample)):
        raise ValueError("policy-plane tensor samples cannot be used with ShogiLegalMovePolicyValueDataset")
    return tensorize_legal_move_policy_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def _policy_plane_value_tensor_sample(
    example: ShogiPolicyValueDatasetItem,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> PolicyPlaneValueTensorSample:
    if isinstance(example, PolicyPlaneValueTensorSample):
        return example
    if isinstance(example, CompactPolicyPlaneValueTensorSample):
        from intrep.representation.outputs.shogi_policy_plane_encoding import SHOGI_POLICY_PLANE_ACTION_COUNT

        policy_plane_targets = torch.zeros(SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.float32)
        policy_plane_targets[example.target_action_indices.long()] = example.target_weights
        policy_plane_legal_mask = torch.zeros(SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.bool)
        policy_plane_legal_mask[example.legal_action_indices.long()] = True
        return PolicyPlaneValueTensorSample(
            position_features=example.position_features,
            policy_plane_targets=policy_plane_targets,
            policy_plane_legal_mask=policy_plane_legal_mask,
            policy_plane_label=example.policy_plane_label,
            value_target=example.value_target,
        )
    if isinstance(example, LegalMovePolicyValueTensorSample):
        raise ValueError("legal-move tensor samples cannot be used with ShogiPolicyPlaneValueDataset")
    return tensorize_policy_plane_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def _compact_policy_plane_value_tensor_sample(
    example: ShogiPolicyValueDatasetItem,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> CompactPolicyPlaneValueTensorSample:
    if isinstance(example, CompactPolicyPlaneValueTensorSample):
        return example
    if isinstance(example, PolicyPlaneValueTensorSample):
        return CompactPolicyPlaneValueTensorSample(
            position_features=example.position_features,
            legal_action_indices=example.policy_plane_legal_mask.nonzero(as_tuple=False).flatten().long(),
            target_action_indices=example.policy_plane_targets.nonzero(as_tuple=False).flatten().long(),
            target_weights=example.policy_plane_targets[example.policy_plane_targets > 0.0].float(),
            policy_plane_label=example.policy_plane_label,
            value_target=example.value_target,
        )
    if isinstance(example, LegalMovePolicyValueTensorSample):
        raise ValueError("legal-move tensor samples cannot be used with ShogiPolicyPlaneValueDataset")
    return tensorize_compact_policy_plane_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def _default_position_features_from_sfen() -> ShogiPositionFeatureBuilder:
    from intrep.representation.inputs.shogi_position_features.position_encoding import shogi_position_features_from_sfen

    return shogi_position_features_from_sfen


def _choice_count(example: ShogiPolicyValueDatasetItem) -> int:
    if isinstance(example, LegalMovePolicyValueTensorSample):
        return int(example.legal_move_features.shape[0])
    if isinstance(example, (PolicyPlaneValueTensorSample, CompactPolicyPlaneValueTensorSample)):
        raise ValueError("policy-plane tensor samples do not have candidate choices")
    return len(example.legal_moves)


def _legal_move_mask(choice_count: int, *, max_legal_move_count: int) -> torch.Tensor:
    legal_move_mask = torch.zeros(max_legal_move_count, dtype=torch.bool)
    legal_move_mask[:choice_count] = True
    return legal_move_mask


def _pad_legal_move_features(legal_move_features: torch.Tensor, *, max_legal_move_count: int) -> torch.Tensor:
    if legal_move_features.shape[0] == max_legal_move_count:
        return legal_move_features
    padded = torch.zeros(
        (max_legal_move_count, legal_move_features.shape[1]),
        dtype=legal_move_features.dtype,
    )
    padded[: legal_move_features.shape[0]] = legal_move_features
    return padded


def _pad_policy_targets(policy_targets: torch.Tensor, *, max_legal_move_count: int) -> torch.Tensor:
    if policy_targets.shape[0] == max_legal_move_count:
        return policy_targets
    padded = torch.zeros(max_legal_move_count, dtype=policy_targets.dtype)
    padded[: policy_targets.shape[0]] = policy_targets
    return padded


def _pad_action_indices(action_indices: torch.Tensor, *, max_action_count: int) -> torch.Tensor:
    if action_indices.shape[0] == max_action_count:
        return action_indices
    padded = torch.zeros(max_action_count, dtype=action_indices.dtype)
    padded[: action_indices.shape[0]] = action_indices
    return padded


def _pad_target_weights(target_weights: torch.Tensor, *, max_action_count: int) -> torch.Tensor:
    if target_weights.shape[0] == max_action_count:
        return target_weights
    padded = torch.zeros(max_action_count, dtype=target_weights.dtype)
    padded[: target_weights.shape[0]] = target_weights
    return padded


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
