import unittest
from unittest.mock import Mock

import torch

from intrep.problems.shogi_policy_value.examples import (
    ShogiLegalMoveTokenPolicyValueDataset,
    collate_legal_move_token_policy_value_samples,
    tensorize_policy_plane_value_examples,
)
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves
from intrep.worlds.shogi.move_encoding import NO_FROM_SQUARE_ID
from intrep.problems.shogi_policy_value.model import (
    DirectCandidateMoveShogiPolicyValueModel,
    DirectCandidateMoveShogiPolicyValueModelConfig,
    PolicyPlaneShogiPolicyValueModel,
    PolicyPlaneShogiPolicyValueModelConfig,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_POSITION_INPUT_MODULE_ID,
    SHOGI_SHARED_CORE_MODULE_ID,
    SHOGI_VALUE_OUTPUT_MODULE_ID,
    SharedCoreShogiPolicyValueModel,
    SharedCoreShogiPolicyValueModelConfig,
    _state_token_hidden,
    shogi_policy_value_model_spec,
)
from intrep.representation.inputs.shogi_position import ShogiPositionGeometryAttentionBias, ShogiPositionInputLayer
from intrep.representation.outputs.shogi_legal_move_token import _legal_move_token_square_hidden
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT
from intrep.worlds.shogi.position_encoding import (
    LINE_TOKEN_OFFSET,
    PAIR_RELATION_PIECE_ON_SQUARE,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SQUARE_TOKEN_OFFSET,
    ShogiPositionFeatures,
    stack_shogi_position_features,
)


class ShogiPolicyValueModelTest(unittest.TestCase):
    def test_policy_plane_model_spec_uses_fixed_policy_head(self) -> None:
        spec = shogi_policy_value_model_spec(
            input=SHOGI_POSITION_INPUT_MODULE_ID,
            core=SHOGI_SHARED_CORE_MODULE_ID,
            policy_output=SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
            value_output=SHOGI_VALUE_OUTPUT_MODULE_ID,
        )

        self.assertEqual(spec["policy_output"], "shogi_policy_plane_policy_output")

    def test_model_returns_legal_move_token_logits(self) -> None:
        position_features, legal_move_token_features, legal_move_token_mask, _, _, _ = _batch()
        model = DirectCandidateMoveShogiPolicyValueModel(DirectCandidateMoveShogiPolicyValueModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_features, legal_move_token_features, legal_move_token_mask)

        self.assertEqual(tuple(logits.shape), tuple(legal_move_token_mask.shape))

    def test_model_masks_invalid_legal_move_tokens(self) -> None:
        position_features, legal_move_token_features, legal_move_token_mask, _, _, _ = _batch()
        legal_move_token_mask[:, -1] = False
        model = DirectCandidateMoveShogiPolicyValueModel(DirectCandidateMoveShogiPolicyValueModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_features, legal_move_token_features, legal_move_token_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_shared_core_model_returns_legal_move_token_logits(self) -> None:
        position_features, legal_move_token_features, legal_move_token_mask, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_features, legal_move_token_features, legal_move_token_mask)

        self.assertEqual(tuple(logits.shape), tuple(legal_move_token_mask.shape))

    def test_shared_core_model_returns_position_value(self) -> None:
        position_features, _, _, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        values = model.predict_value(position_features)

        self.assertEqual(tuple(values.shape), (2,))
        self.assertLessEqual(float(values.abs().max().item()), 1.0)

    def test_shared_core_model_returns_policy_and_value_with_one_core_forward(self) -> None:
        position_features, legal_move_token_features, legal_move_token_mask, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )
        model.encoder.core.forward = Mock(wraps=model.encoder.core.forward)

        logits, values = model.forward_policy_value(position_features, legal_move_token_features, legal_move_token_mask)

        self.assertEqual(tuple(logits.shape), tuple(legal_move_token_mask.shape))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(model.encoder.core.forward.call_count, 1)

    def test_shared_core_policy_head_scores_legal_move_tokens_after_cross_attention(self) -> None:
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        self.assertEqual(model.policy_output.policy_head.scorer[0].in_features, 8 * 2)

    def test_legal_move_token_square_hidden_maps_square_ids_to_board_tokens(self) -> None:
        position_hidden = torch.arange(2 * SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT * 3, dtype=torch.float32).reshape(
            2,
            SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
            3,
        )
        square_ids = torch.tensor([[0, 80], [NO_FROM_SQUARE_ID, 7]])

        square_hidden = _legal_move_token_square_hidden(
            position_hidden,
            square_ids,
            zero_square_id=NO_FROM_SQUARE_ID,
        )

        self.assertTrue(torch.equal(square_hidden[0, 0], position_hidden[0, SQUARE_TOKEN_OFFSET]))
        self.assertTrue(torch.equal(square_hidden[0, 1], position_hidden[0, SQUARE_TOKEN_OFFSET + 80]))
        self.assertTrue(torch.equal(square_hidden[1, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(square_hidden[1, 1], position_hidden[1, SQUARE_TOKEN_OFFSET + 7]))

    def test_position_input_layer_builds_global_square_piece_sequence(self) -> None:
        position_features, _, _, _, _, _ = _batch()
        layer = ShogiPositionInputLayer(embedding_dim=8)

        embeddings = layer(position_features)

        self.assertEqual(tuple(embeddings.shape), (2, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 8))
        self.assertFalse(hasattr(layer, "piece_slot_embedding"))

    def test_position_input_layer_normalizes_feature_groups(self) -> None:
        position_features, _, _, _, _, _ = _batch()
        layer = ShogiPositionInputLayer(embedding_dim=8)

        embeddings = layer(position_features)

        token_norms = embeddings.norm(dim=-1)
        self.assertLess(float(token_norms.max().item() - token_norms.min().item()), 1e-3)

    def test_state_token_hidden_uses_first_position_token(self) -> None:
        position_hidden = torch.arange(2 * SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT * 3, dtype=torch.float32).reshape(
            2,
            SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
            3,
        )

        state_hidden = _state_token_hidden(position_hidden)

        self.assertTrue(torch.equal(state_hidden, position_hidden[:, 0]))

    def test_position_geometry_attention_bias_targets_square_and_line_pairs(self) -> None:
        position_features, _, _, _, _, _ = _batch()
        embeddings = torch.zeros((2, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 8))
        attention_bias = ShogiPositionGeometryAttentionBias()
        attention_bias.relation_bias.weight.data[:, 0] = torch.arange(17 * 17, dtype=torch.float32)
        attention_bias.line_square_relation_bias.weight.data[:, 0] = torch.tensor([0.0, 500.0])

        bias = attention_bias(position_features, embeddings)
        same_square_relation = 8 * 17 + 8
        one_file_right_relation = 8 * 17 + 9

        self.assertEqual(
            tuple(bias.shape),
            (2, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT),
        )
        self.assertEqual(float(bias[0, SQUARE_TOKEN_OFFSET, SQUARE_TOKEN_OFFSET].item()), float(same_square_relation))
        self.assertEqual(
            float(bias[0, SQUARE_TOKEN_OFFSET, SQUARE_TOKEN_OFFSET + 1].item()),
            float(one_file_right_relation),
        )
        self.assertEqual(float(bias[0, 0, SQUARE_TOKEN_OFFSET].item()), 0.0)
        self.assertEqual(float(bias[0, SQUARE_TOKEN_OFFSET, 0].item()), 0.0)
        self.assertEqual(float(bias[0, LINE_TOKEN_OFFSET, SQUARE_TOKEN_OFFSET].item()), 500.0)
        self.assertEqual(float(bias[0, SQUARE_TOKEN_OFFSET, LINE_TOKEN_OFFSET].item()), 500.0)
        self.assertEqual(float(bias[0, LINE_TOKEN_OFFSET, SQUARE_TOKEN_OFFSET + 1].item()), 0.0)

    def test_position_geometry_attention_bias_adds_dynamic_pair_relation_bias(self) -> None:
        position_features, _, _, _, _, _ = _batch()
        embeddings = torch.zeros((2, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, 8))
        attention_bias = ShogiPositionGeometryAttentionBias()
        attention_bias.pair_relation_bias.weight.data[:, 0] = 0.0
        attention_bias.pair_relation_bias.weight.data[PAIR_RELATION_PIECE_ON_SQUARE, 0] = 700.0

        bias = attention_bias(position_features, embeddings)
        pair_relation_edges = position_features.pair_relation_edges
        relation_indices = pair_relation_edges.relation_ids.eq(PAIR_RELATION_PIECE_ON_SQUARE).nonzero()

        self.assertGreater(int(relation_indices.size(0)), 0)
        edge_index = int(relation_indices[0].item())
        batch = int(pair_relation_edges.batch_indices[edge_index].item())
        row = int(pair_relation_edges.source_token_indices[edge_index].item())
        column = int(pair_relation_edges.target_token_indices[edge_index].item())
        self.assertEqual(float(bias[batch, row, column].item()), 700.0)

    def test_shared_models_pass_position_geometry_attention_bias_to_core(self) -> None:
        position_features, legal_move_token_features, legal_move_token_mask, _, _, _ = _batch()
        model = SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )
        model.encoder.core.forward = Mock(wraps=model.encoder.core.forward)

        model(position_features, legal_move_token_features, legal_move_token_mask)

        self.assertIn("attention_bias", model.encoder.core.forward.call_args.kwargs)

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
        position_features, legal_action_mask = _policy_plane_batch()
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_features, legal_action_mask)

        self.assertEqual(tuple(logits.shape), (2, SHOGI_POLICY_PLANE_ACTION_COUNT))

    def test_policy_plane_model_masks_illegal_actions(self) -> None:
        position_features, legal_action_mask = _policy_plane_batch()
        legal_action_mask[:, -1] = False
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        logits = model(position_features, legal_action_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_policy_plane_model_returns_policy_and_value_with_one_core_forward(self) -> None:
        position_features, legal_action_mask = _policy_plane_batch()
        model = PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )
        model.encoder.core.forward = Mock(wraps=model.encoder.core.forward)

        logits, values = model.forward_policy_value(position_features, legal_action_mask)

        self.assertEqual(tuple(logits.shape), (2, SHOGI_POLICY_PLANE_ACTION_COUNT))
        self.assertEqual(tuple(values.shape), (2,))
        self.assertEqual(model.encoder.core.forward.call_count, 1)


def _batch() -> tuple[ShogiPositionFeatures, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
    dataset = ShogiLegalMoveTokenPolicyValueDataset(examples)
    batch = collate_legal_move_token_policy_value_samples([dataset[index] for index in range(len(dataset))])
    return (
        batch.position_features,
        batch.legal_move_token_features,
        batch.legal_move_token_mask,
        batch.labels,
        batch.policy_targets,
        batch.value_targets,
    )


def _policy_plane_batch() -> tuple[ShogiPositionFeatures, torch.Tensor]:
    examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
    samples = tensorize_policy_plane_value_examples(examples)
    return (
        stack_shogi_position_features([sample.position_features for sample in samples]),
        torch.stack([sample.policy_plane_legal_mask for sample in samples]),
    )


if __name__ == "__main__":
    unittest.main()
