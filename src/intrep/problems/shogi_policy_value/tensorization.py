from __future__ import annotations

from typing import Callable, Sequence

import shogi

from intrep.problems.shogi_policy_value.examples import (
    ShogiMoveChoiceExample,
    ShogiMovePolicyValueExample,
)
from intrep.problems.shogi_policy_value.samples import (
    CompactActionPlanePolicyValueTensorSample,
    LegalMovePolicyValueTensorSample,
    ActionPlanePolicyValueTensorSample,
    ShogiPolicyValueDatasetItem,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures

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
    position_features, legal_move_feature_ids, _legal_move_mask, move_index, policy_targets = policy_sample(
        example,
        max_legal_move_count=len(example.legal_moves),
        position_features_from_sfen=position_features_from_sfen,
    )
    return LegalMovePolicyValueTensorSample(
        position_features=position_features,
        legal_move_feature_ids=legal_move_feature_ids,
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


def tensorize_action_plane_policy_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> ActionPlanePolicyValueTensorSample:
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi action-plane policy samples")
    from intrep.representation.outputs.shogi_action_plane_policy_encoding import (
        SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT,
        shogi_action_plane_policy_action_index,
        shogi_action_plane_policy_legal_mask,
    )
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    action_plane_policy_label = torch.tensor(
        shogi_action_plane_policy_action_index(example.chosen_move, turn=board.turn),
        dtype=torch.long,
    )
    action_plane_policy_targets = torch.zeros(SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT, dtype=torch.float32)
    if example.policy_targets is None:
        if example.policy_target_source != "chosen_move":
            raise ValueError(f"missing policy_targets for policy_target_source={example.policy_target_source}")
        action_plane_policy_targets[int(action_plane_policy_label.item())] = 1.0
    else:
        total = sum(example.policy_targets.values())
        for move, weight in example.policy_targets.items():
            action_index = shogi_action_plane_policy_action_index(move, turn=board.turn)
            action_plane_policy_targets[action_index] = float(weight) / total
    return ActionPlanePolicyValueTensorSample(
        position_features=position_features,
        action_plane_policy_targets=action_plane_policy_targets,
        action_plane_policy_legal_mask=shogi_action_plane_policy_legal_mask(board),
        action_plane_policy_label=action_plane_policy_label,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_action_plane_policy_value_examples(
    examples: Sequence[ShogiMovePolicyValueExample],
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> list[ActionPlanePolicyValueTensorSample]:
    return [
        tensorize_action_plane_policy_value_example(
            example,
            position_features_from_sfen=position_features_from_sfen,
        )
        for example in examples
    ]


def tensorize_compact_action_plane_policy_value_example(
    example: ShogiMovePolicyValueExample,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> CompactActionPlanePolicyValueTensorSample:
    if torch is None:
        raise RuntimeError("torch is required to materialize shogi action-plane policy samples")
    from intrep.representation.outputs.shogi_action_plane_policy_encoding import shogi_action_plane_policy_action_index
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    legal_action_indices = torch.tensor(
        [shogi_action_plane_policy_action_index(move, turn=board.turn) for move in example.legal_moves],
        dtype=torch.long,
    )
    action_plane_policy_label = torch.tensor(
        shogi_action_plane_policy_action_index(example.chosen_move, turn=board.turn),
        dtype=torch.long,
    )
    if example.policy_targets is None:
        if example.policy_target_source != "chosen_move":
            raise ValueError(f"missing policy_targets for policy_target_source={example.policy_target_source}")
        target_action_indices = action_plane_policy_label.reshape(1)
        target_weights = torch.ones(1, dtype=torch.float32)
    else:
        total = sum(example.policy_targets.values())
        target_action_indices = torch.tensor(
            [shogi_action_plane_policy_action_index(move, turn=board.turn) for move in example.policy_targets],
            dtype=torch.long,
        )
        target_weights = torch.tensor(
            [float(weight) / total for weight in example.policy_targets.values()],
            dtype=torch.float32,
        )
    return CompactActionPlanePolicyValueTensorSample(
        position_features=position_features,
        legal_action_indices=legal_action_indices,
        target_action_indices=target_action_indices,
        target_weights=target_weights,
        action_plane_policy_label=action_plane_policy_label,
        value_target=torch.tensor(
            float("nan") if example.value_target is None else example.value_target,
            dtype=torch.float32,
        ),
    )


def tensorize_compact_action_plane_policy_value_examples(
    examples: Sequence[ShogiMovePolicyValueExample],
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> list[CompactActionPlanePolicyValueTensorSample]:
    return [
        tensorize_compact_action_plane_policy_value_example(
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
    from intrep.representation.outputs.shogi_legal_move_encoding import shogi_legal_move_feature_ids
    board = shogi.Board(example.position_sfen)
    position_features_from_sfen = position_features_from_sfen or _default_position_features_from_sfen()
    position_features = position_features_from_sfen(example.position_sfen)
    legal_move_feature_ids = shogi_legal_move_feature_ids(
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
        legal_move_feature_ids,
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
    if isinstance(example, (ActionPlanePolicyValueTensorSample, CompactActionPlanePolicyValueTensorSample)):
        raise ValueError("action-plane policy tensor samples cannot be used with ShogiLegalMovePolicyValueDataset")
    return tensorize_legal_move_policy_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def action_plane_policy_value_tensor_sample(
    example: ShogiPolicyValueDatasetItem,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> ActionPlanePolicyValueTensorSample:
    if isinstance(example, ActionPlanePolicyValueTensorSample):
        return example
    if isinstance(example, CompactActionPlanePolicyValueTensorSample):
        from intrep.representation.outputs.shogi_action_plane_policy_encoding import SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT

        action_plane_policy_targets = torch.zeros(SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT, dtype=torch.float32)
        action_plane_policy_targets[example.target_action_indices.long()] = example.target_weights
        action_plane_policy_legal_mask = torch.zeros(SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT, dtype=torch.bool)
        action_plane_policy_legal_mask[example.legal_action_indices.long()] = True
        return ActionPlanePolicyValueTensorSample(
            position_features=example.position_features,
            action_plane_policy_targets=action_plane_policy_targets,
            action_plane_policy_legal_mask=action_plane_policy_legal_mask,
            action_plane_policy_label=example.action_plane_policy_label,
            value_target=example.value_target,
        )
    if isinstance(example, LegalMovePolicyValueTensorSample):
        raise ValueError("legal-move tensor samples cannot be used with ShogiActionPlanePolicyValueDataset")
    return tensorize_action_plane_policy_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def compact_action_plane_policy_value_tensor_sample(
    example: ShogiPolicyValueDatasetItem,
    *,
    position_features_from_sfen: ShogiPositionFeatureBuilder | None = None,
) -> CompactActionPlanePolicyValueTensorSample:
    if isinstance(example, CompactActionPlanePolicyValueTensorSample):
        return example
    if isinstance(example, ActionPlanePolicyValueTensorSample):
        return CompactActionPlanePolicyValueTensorSample(
            position_features=example.position_features,
            legal_action_indices=example.action_plane_policy_legal_mask.nonzero(as_tuple=False).flatten().long(),
            target_action_indices=example.action_plane_policy_targets.nonzero(as_tuple=False).flatten().long(),
            target_weights=example.action_plane_policy_targets[example.action_plane_policy_targets > 0.0].float(),
            action_plane_policy_label=example.action_plane_policy_label,
            value_target=example.value_target,
        )
    if isinstance(example, LegalMovePolicyValueTensorSample):
        raise ValueError("legal-move tensor samples cannot be used with ShogiActionPlanePolicyValueDataset")
    return tensorize_compact_action_plane_policy_value_example(
        example,
        position_features_from_sfen=position_features_from_sfen,
    )


def choice_count(example: ShogiPolicyValueDatasetItem) -> int:
    if isinstance(example, LegalMovePolicyValueTensorSample):
        return int(example.legal_move_feature_ids.shape[0])
    if isinstance(example, (ActionPlanePolicyValueTensorSample, CompactActionPlanePolicyValueTensorSample)):
        raise ValueError("action-plane policy tensor samples do not have candidate choices")
    return len(example.legal_moves)


def _default_position_features_from_sfen() -> ShogiPositionFeatureBuilder:
    from intrep.representation.inputs.shogi_position_features.position_rich import shogi_rich_position_features_from_sfen

    return shogi_rich_position_features_from_sfen
