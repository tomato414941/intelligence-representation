import tempfile
import unittest
from pathlib import Path

import torch

from intrep.shogi_move_choice import ShogiMoveChoiceDataset, shogi_move_choice_examples_from_usi_moves
from intrep.shogi_move_choice_checkpoint import load_shogi_move_choice_checkpoint, save_shogi_move_choice_checkpoint
from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingConfig, train_shogi_move_choice_model


class ShogiMoveChoiceCheckpointTest(unittest.TestCase):
    def test_save_and_load_preserves_logits(self) -> None:
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))
        result = train_shogi_move_choice_model(
            examples,
            config=ShogiMoveChoiceTrainingConfig(
                max_steps=2,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )
        position_token_ids, candidate_move_features, candidate_mask, _ = next(
            iter(torch.utils.data.DataLoader(ShogiMoveChoiceDataset(examples), batch_size=2))
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi.pt"
            save_shogi_move_choice_checkpoint(path, result)
            loaded = load_shogi_move_choice_checkpoint(path)

        with torch.no_grad():
            expected = result.model(position_token_ids, candidate_move_features, candidate_mask)
            actual = loaded(position_token_ids, candidate_move_features, candidate_mask)

        self.assertTrue(torch.allclose(actual, expected))


if __name__ == "__main__":
    unittest.main()
