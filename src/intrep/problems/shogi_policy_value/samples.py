from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intrep.problems.shogi_policy_value.examples import ShogiMovePolicyValueExample
from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPositionFeatures,
    stack_shogi_position_features,
)

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised in lightweight preprocessing environments.
    torch = None
    TorchDataset = object


@dataclass(frozen=True)
class LegalMovePolicyValueTensorSample:
    position_features: ShogiPositionFeatures
    legal_move_feature_ids: torch.Tensor
    label: torch.Tensor
    policy_targets: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class ActionPlanePolicyValueTensorSample:
    position_features: ShogiPositionFeatures
    action_plane_policy_targets: torch.Tensor
    action_plane_policy_legal_mask: torch.Tensor
    action_plane_policy_label: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class CompactActionPlanePolicyValueTensorSample:
    position_features: ShogiPositionFeatures
    legal_action_indices: torch.Tensor
    target_action_indices: torch.Tensor
    target_weights: torch.Tensor
    action_plane_policy_label: torch.Tensor
    value_target: torch.Tensor


@dataclass(frozen=True)
class LegalMovePolicyValueBatch:
    position_features: ShogiPositionFeatures
    legal_move_feature_ids: torch.Tensor
    legal_move_mask: torch.Tensor
    labels: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor

    def to(self, device: torch.device) -> "LegalMovePolicyValueBatch":
        return LegalMovePolicyValueBatch(
            position_features=self.position_features.to(device),
            legal_move_feature_ids=self.legal_move_feature_ids.to(device),
            legal_move_mask=self.legal_move_mask.to(device),
            labels=self.labels.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
        )


@dataclass(frozen=True)
class ActionPlanePolicyValueBatch:
    position_features: ShogiPositionFeatures
    legal_action_mask: torch.Tensor
    labels: torch.Tensor
    target_action_indices: torch.Tensor
    target_weights: torch.Tensor
    value_targets: torch.Tensor

    def to(self, device: torch.device) -> "ActionPlanePolicyValueBatch":
        return ActionPlanePolicyValueBatch(
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
    | ActionPlanePolicyValueTensorSample
    | CompactActionPlanePolicyValueTensorSample
)


class ShogiLegalMovePolicyValueDataset(TorchDataset):
    def __init__(
        self,
        examples: Sequence[ShogiPolicyValueDatasetItem],
        *,
        position_features_from_sfen=None,
    ) -> None:
        from intrep.problems.shogi_policy_value.tensorization import choice_count

        if not examples:
            raise ValueError("examples must not be empty")
        self.examples = examples
        self.position_features_from_sfen = position_features_from_sfen
        self.max_legal_move_count = int(
            getattr(examples, "max_legal_move_count", None) or max(choice_count(example) for example in self.examples)
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiLegalMovePolicyValueDataset items")
        from intrep.problems.shogi_policy_value.tensorization import legal_move_policy_value_tensor_sample

        return legal_move_policy_value_tensor_sample(
            self.examples[index],
            position_features_from_sfen=self.position_features_from_sfen,
        )


class ShogiActionPlanePolicyValueDataset(TorchDataset):
    def __init__(
        self,
        examples: Sequence[ShogiPolicyValueDatasetItem],
        *,
        position_features_from_sfen=None,
    ) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        if any(isinstance(example, LegalMovePolicyValueTensorSample) for example in examples):
            raise ValueError("legal-move tensor samples cannot be used with ShogiActionPlanePolicyValueDataset")
        self.examples = examples
        self.position_features_from_sfen = position_features_from_sfen

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        if torch is None:
            raise RuntimeError("torch is required to materialize ShogiActionPlanePolicyValueDataset items")
        from intrep.problems.shogi_policy_value.tensorization import compact_action_plane_policy_value_tensor_sample

        return compact_action_plane_policy_value_tensor_sample(
            self.examples[index],
            position_features_from_sfen=self.position_features_from_sfen,
        )


def collate_legal_move_policy_value_samples(
    samples: Sequence[LegalMovePolicyValueTensorSample],
) -> LegalMovePolicyValueBatch:
    if torch is None:
        raise RuntimeError("torch is required to collate shogi policy/value samples")
    max_legal_move_count = max(int(sample.legal_move_feature_ids.shape[0]) for sample in samples)
    return LegalMovePolicyValueBatch(
        position_features=stack_shogi_position_features([sample.position_features for sample in samples]),
        legal_move_feature_ids=torch.stack(
            [
                _pad_legal_move_feature_ids(sample.legal_move_feature_ids, max_legal_move_count=max_legal_move_count)
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


def collate_action_plane_policy_value_samples(
    samples: Sequence[CompactActionPlanePolicyValueTensorSample],
) -> ActionPlanePolicyValueBatch:
    if torch is None:
        raise RuntimeError("torch is required to collate shogi action-plane policy samples")
    from intrep.representation.outputs.shogi_action_plane_policy_encoding import SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT

    max_target_count = max(int(sample.target_action_indices.shape[0]) for sample in samples)
    legal_action_mask = torch.zeros((len(samples), SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT), dtype=torch.bool)
    for row_index, sample in enumerate(samples):
        legal_action_mask[row_index, sample.legal_action_indices.long()] = True
    return ActionPlanePolicyValueBatch(
        position_features=stack_shogi_position_features([sample.position_features for sample in samples]),
        legal_action_mask=legal_action_mask,
        labels=torch.stack([sample.action_plane_policy_label for sample in samples]),
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


def _choice_count(example: LegalMovePolicyValueTensorSample) -> int:
    return int(example.legal_move_feature_ids.shape[0])


def _legal_move_mask(choice_count: int, *, max_legal_move_count: int) -> torch.Tensor:
    legal_move_mask = torch.zeros(max_legal_move_count, dtype=torch.bool)
    legal_move_mask[:choice_count] = True
    return legal_move_mask


def _pad_legal_move_feature_ids(legal_move_feature_ids: torch.Tensor, *, max_legal_move_count: int) -> torch.Tensor:
    if legal_move_feature_ids.shape[0] == max_legal_move_count:
        return legal_move_feature_ids
    padded = torch.zeros(
        (max_legal_move_count, legal_move_feature_ids.shape[1]),
        dtype=legal_move_feature_ids.dtype,
    )
    padded[: legal_move_feature_ids.shape[0]] = legal_move_feature_ids
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
