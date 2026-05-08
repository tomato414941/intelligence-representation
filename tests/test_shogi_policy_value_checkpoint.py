import tempfile
import unittest
from pathlib import Path

import torch

from intrep.tasks.shogi_policy_value.examples import ShogiPolicyValueDataset
from tests.shogi_test_helpers import shogi_policy_value_examples_from_test_moves
from intrep.tasks.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint, save_shogi_policy_value_checkpoint
from intrep.tasks.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, train_shogi_policy_value_model


class ShogiPolicyValueCheckpointTest(unittest.TestCase):
    def test_save_and_load_preserves_logits(self) -> None:
        examples = shogi_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=2,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )
        position_token_ids, candidate_move_features, candidate_mask, _, _, _ = next(
            iter(torch.utils.data.DataLoader(ShogiPolicyValueDataset(examples), batch_size=2))
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi.pt"
            save_shogi_policy_value_checkpoint(path, result)
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(position_token_ids, candidate_move_features, candidate_mask)
            actual = loaded(position_token_ids, candidate_move_features, candidate_mask)

        self.assertTrue(torch.allclose(actual, expected))

    def test_load_rejects_missing_model_weights(self) -> None:
        examples = shogi_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi.pt"
            save_shogi_policy_value_checkpoint(path, result)
            payload = torch.load(path, weights_only=False)
            payload["model_state_dict"].pop(next(iter(payload["model_state_dict"])))
            torch.save(payload, path)

            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_shogi_policy_value_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
