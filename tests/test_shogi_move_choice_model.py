import unittest
from unittest.mock import Mock

import torch

from intrep.tasks.shogi_move_choice.examples import ShogiPolicyValueDataset
from tests.shogi_test_helpers import shogi_policy_value_examples_from_test_moves
from intrep.worlds.shogi.move_encoding import NO_FROM_SQUARE_ID
from intrep.tasks.shogi_move_choice.model import (
    SharedCoreShogiMoveChoiceModel,
    SharedCoreShogiMoveChoiceModelConfig,
    ShogiMoveChoiceModel,
    ShogiMoveChoiceModelConfig,
    _candidate_square_hidden,
)
from intrep.worlds.shogi.position_encoding import SHOGI_POSITION_TOKEN_COUNT


class ShogiMoveChoiceModelTest(unittest.TestCase):
    def test_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        model = ShogiMoveChoiceModel(ShogiMoveChoiceModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertEqual(tuple(logits.shape), tuple(candidate_mask.shape))

    def test_model_masks_invalid_candidates(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        candidate_mask[:, -1] = False
        model = ShogiMoveChoiceModel(ShogiMoveChoiceModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_shared_core_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = _batch()
        model = SharedCoreShogiMoveChoiceModel(
            SharedCoreShogiMoveChoiceModelConfig(
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
        model = SharedCoreShogiMoveChoiceModel(
            SharedCoreShogiMoveChoiceModelConfig(
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
        model = SharedCoreShogiMoveChoiceModel(
            SharedCoreShogiMoveChoiceModelConfig(
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

    def test_shared_core_candidate_scorer_uses_position_and_square_hidden(self) -> None:
        model = SharedCoreShogiMoveChoiceModel(
            SharedCoreShogiMoveChoiceModelConfig(
                embedding_dim=8,
                num_heads=2,
                hidden_dim=16,
                num_layers=1,
            )
        )

        self.assertEqual(model.candidate_scorer[0].in_features, 8 * 4)

    def test_candidate_square_hidden_maps_square_ids_to_board_tokens(self) -> None:
        position_hidden = torch.arange(2 * SHOGI_POSITION_TOKEN_COUNT * 3, dtype=torch.float32).reshape(
            2,
            SHOGI_POSITION_TOKEN_COUNT,
            3,
        )
        square_ids = torch.tensor([[0, 80], [NO_FROM_SQUARE_ID, 7]])

        square_hidden = _candidate_square_hidden(
            position_hidden,
            square_ids,
            zero_square_id=NO_FROM_SQUARE_ID,
        )

        self.assertTrue(torch.equal(square_hidden[0, 0], position_hidden[0, 1]))
        self.assertTrue(torch.equal(square_hidden[0, 1], position_hidden[0, 81]))
        self.assertTrue(torch.equal(square_hidden[1, 0], torch.zeros(3)))
        self.assertTrue(torch.equal(square_hidden[1, 1], position_hidden[1, 8]))


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = shogi_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
    dataset = ShogiPolicyValueDataset(examples)
    rows = [dataset[index] for index in range(len(dataset))]
    return tuple(torch.stack(values) for values in zip(*rows))  # type: ignore[return-value]


if __name__ == "__main__":
    unittest.main()
