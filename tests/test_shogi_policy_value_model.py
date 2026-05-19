import unittest
from unittest.mock import Mock

import torch

from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyValueDataset,
    collate_candidate_move_policy_value_samples,
    tensorize_policy_plane_value_examples,
)
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves
from intrep.worlds.shogi.move_encoding import NO_FROM_SQUARE_ID
from intrep.problems.shogi_policy_value.model import (
    DirectShogiPolicyValueModel,
    DirectShogiPolicyValueModelConfig,
    PolicyPlaneShogiPolicyValueModel,
    PolicyPlaneShogiPolicyValueModelConfig,
    SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC,
    SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
    ShogiPositionInputLayer,
    ShogiPolicyPlaneHead,
    SharedCoreShogiPolicyValueModel,
    SharedCoreShogiPolicyValueModelConfig,
    _candidate_square_hidden,
    shogi_policy_value_model_spec,
)
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT
from intrep.worlds.shogi.position_encoding import SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, SQUARE_TOKEN_OFFSET


class ShogiPolicyValueModelTest(unittest.TestCase):
    def test_policy_plane_model_spec_uses_fixed_policy_head(self) -> None:
        spec = shogi_policy_value_model_spec(SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER)

        self.assertEqual(spec, SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC)
        self.assertIsNone(spec["candidate_move_input"])

    def test_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        model = DirectShogiPolicyValueModel(DirectShogiPolicyValueModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertEqual(tuple(logits.shape), tuple(candidate_mask.shape))

    def test_model_masks_invalid_candidates(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        candidate_mask[:, -1] = False
        model = DirectShogiPolicyValueModel(DirectShogiPolicyValueModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_shared_core_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertEqual(tuple(logits.shape), tuple(candidate_mask.shape))

    def test_shared_core_model_returns_position_value(self) -> None:
        position_token_ids, _, _, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        values = model.predict_value(position_token_ids)

        self.assertEqual(tuple(values.shape), (2,))
        self.assertLessEqual(float(values.abs().max().item()), 1.0)

    def test_shared_core_model_returns_policy_and_value_with_one_core_forward(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )
        model.core.forward = Mock(wraps=model.core.forward)

        logits, values = model.forward_policy_value(position_token_ids, candidate_move_features, candidate_mask)

        self.assertEqual(tuple(logits.shape), tuple(candidate_mask.shape))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(model.core.forward.call_count, 1)

    def test_shared_core_policy_head_uses_position_move_and_square_hidden(self) -> None:
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        self.assertEqual(model.policy_head.scorer[0].in_features, 8 * 4)

    def test_candidate_square_hidden_maps_square_ids_to_board_tokens(self) -> None:
        position_hidden = torch.arange(2 * SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT * 3, dtype=torch.float32).reshape(
            2,
            SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
            3,
        )
        square_ids = torch.tensor([[0, 80], [NO_FROM_SQUARE_ID, 7]])

        square_hidden = _candidate_square_hidden(
            position_hidden,
            square_ids,
            zero_square_id=NO_FROM_SQUARE_ID,
        )

        self.assertTrue(torch.equal(square_hidden[0, 0], position_hidden[0, SQUARE_TOKEN_OFFSET]))
        self.assertTrue(torch.equal(square_hidden[0, 1], position_hidden[0, SQUARE_TOKEN_OFFSET + 80]))
        self.assertTrue(torch.equal(square_hidden[1, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(square_hidden[1, 1], position_hidden[1, SQUARE_TOKEN_OFFSET + 7]))

    def test_position_input_layer_builds_global_square_sequence(self) -> None:
        position_token_ids, _, _, _, _, _ = _batch()
        layer = ShogiPositionInputLayer(embedding_dim=8)

        embeddings = layer(position_token_ids)

        self.assertEqual(tuple(embeddings.shape), (2, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 8))

    def test_policy_plane_head_returns_fixed_action_logits(self) -> None:
        head = ShogiPolicyPlaneHead(
            embedding_dim=8,
            hidden_dim=16,
            action_count=SHOGI_POLICY_PLANE_ACTION_COUNT,
        )
        position_embedding = torch.randn(2, 8)
        legal_action_mask = torch.ones(2, SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.bool)

        logits = head(position_embedding, legal_action_mask)

        self.assertEqual(tuple(logits.shape), (2, SHOGI_POLICY_PLANE_ACTION_COUNT))

    def test_policy_plane_head_masks_illegal_actions(self) -> None:
        head = ShogiPolicyPlaneHead(
            embedding_dim=8,
            hidden_dim=16,
            action_count=SHOGI_POLICY_PLANE_ACTION_COUNT,
        )
        position_embedding = torch.randn(2, 8)
        legal_action_mask = torch.ones(2, SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.bool)
        legal_action_mask[:, -1] = False

        logits = head(position_embedding, legal_action_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)
        self.assertLess(float(logits[1, -1].item()), -1e20)

    def test_policy_plane_model_returns_fixed_action_logits(self) -> None:
        position_token_ids, legal_action_mask = _policy_plane_batch()
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_token_ids, legal_action_mask)

        self.assertEqual(tuple(logits.shape), (2, SHOGI_POLICY_PLANE_ACTION_COUNT))

    def test_policy_plane_model_masks_illegal_actions(self) -> None:
        position_token_ids, legal_action_mask = _policy_plane_batch()
        legal_action_mask[:, -1] = False
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_token_ids, legal_action_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_policy_plane_model_returns_policy_and_value_with_one_core_forward(self) -> None:
        position_token_ids, legal_action_mask = _policy_plane_batch()
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )
        model.core.forward = Mock(wraps=model.core.forward)

        logits, values = model.forward_policy_value(position_token_ids, legal_action_mask)

        self.assertEqual(tuple(logits.shape), (2, SHOGI_POLICY_PLANE_ACTION_COUNT))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(model.core.forward.call_count, 1)


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
    dataset = ShogiPolicyValueDataset(examples)
    batch = collate_candidate_move_policy_value_samples([dataset[index] for index in range(len(dataset))])
    return (
        batch.position_token_ids,
        batch.candidate_move_features,
        batch.candidate_mask,
        batch.labels,
        batch.policy_targets,
        batch.value_targets,
    )


def _policy_plane_batch() -> tuple[torch.Tensor, torch.Tensor]:
    examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
    samples = tensorize_policy_plane_value_examples(examples)
    return (
        torch.stack([sample.position_token_ids for sample in samples]),
        torch.stack([sample.policy_plane_legal_mask for sample in samples]),
    )


if __name__ == "__main__":
    unittest.main()
