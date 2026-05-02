import unittest

import torch

from intrep.shogi_move_choice import ShogiMoveChoiceDataset, shogi_move_choice_examples_from_usi_moves
from intrep.shogi_move_choice_model import (
    SharedCoreShogiMoveChoiceModel,
    SharedCoreShogiMoveChoiceModelConfig,
    ShogiMoveChoiceModel,
    ShogiMoveChoiceModelConfig,
)


class ShogiMoveChoiceModelTest(unittest.TestCase):
    def test_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _ = _batch()
        model = ShogiMoveChoiceModel(ShogiMoveChoiceModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertEqual(tuple(logits.shape), tuple(candidate_mask.shape))

    def test_model_masks_invalid_candidates(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _ = _batch()
        candidate_mask[:, -1] = False
        model = ShogiMoveChoiceModel(ShogiMoveChoiceModelConfig(embedding_dim=8, hidden_dim=16))

        logits = model(position_token_ids, candidate_move_features, candidate_mask)

        self.assertLess(float(logits[0, -1].item()), -1e20)

    def test_shared_core_model_returns_candidate_logits(self) -> None:
        position_token_ids, candidate_move_features, candidate_mask, _ = _batch()
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


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))
    dataset = ShogiMoveChoiceDataset(examples)
    rows = [dataset[index] for index in range(len(dataset))]
    return tuple(torch.stack(values) for values in zip(*rows))  # type: ignore[return-value]


if __name__ == "__main__":
    unittest.main()
