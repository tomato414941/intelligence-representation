from __future__ import annotations

from typing import Callable, Sequence

import shogi

from intrep.problems.shogi_policy_value.examples import (
    ShogiMoveChoiceExample,
    ShogiMovePolicyValueExample,
)
from intrep.problems.shogi_policy_value.samples import (
    CompactPolicyPlaneValueTensorSample,
    LegalMovePolicyValueTensorSample,
    PolicyPlaneValueTensorSample,
    ShogiPolicyValueDatasetItem,
)
from intrep.representation.inputs.shogi_position_features.position_encoding import ShogiPositionFeatures

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised in lightweight preprocessing environments.
    torch = None
    TorchDataset = object


ShogiPositionFeatureBuilder = Callable[[str], ShogiPositionFeatures]


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
        return policy_sample(
            self.examples[index],
            max_legal_move_count=self.max_legal_move_count,
            position_features_from_sfen=self.position_features_from_sfen,
        )


def tensorize_legal_move_policy_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> LegalMovePolicyValueTensorSample:
    position_features, legal_move_features, _legal_move_mask, move_index, policy_targets = policy_sample(
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


def policy_sample(
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


def legal_move_policy_value_tensor_sample(
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


def policy_plane_value_tensor_sample(
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


def compact_policy_plane_value_tensor_sample(
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


def choice_count(example: ShogiPolicyValueDatasetItem) -> int:
    if isinstance(example, LegalMovePolicyValueTensorSample):
        return int(example.legal_move_features.shape[0])
    if isinstance(example, (PolicyPlaneValueTensorSample, CompactPolicyPlaneValueTensorSample)):
        raise ValueError("policy-plane tensor samples do not have candidate choices")
    return len(example.legal_moves)


def _default_position_features_from_sfen() -> ShogiPositionFeatureBuilder:
    from intrep.representation.inputs.shogi_position_features.position_encoding import shogi_position_features_from_sfen

    return shogi_position_features_from_sfen
