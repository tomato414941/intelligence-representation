import tempfile
import unittest
from pathlib import Path

import torch

from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyPlaneValueDataset,
    ShogiPolicyValueDataset,
    collate_candidate_move_policy_value_samples,
    collate_policy_plane_value_samples,
)
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves
from intrep.problems.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint, save_shogi_policy_value_checkpoint
from intrep.problems.shogi_policy_value.model import (
    SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC,
    SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
    SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER,
    SHOGI_POLICY_VALUE_MODEL_SPEC,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, train_shogi_policy_value_model


class ShogiPolicyValueCheckpointTest(unittest.TestCase):
    def test_save_and_load_preserves_logits(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
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
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    ShogiPolicyValueDataset(examples),
                    batch_size=2,
                    collate_fn=collate_candidate_move_policy_value_samples,
                )
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi.pt"
            save_shogi_policy_value_checkpoint(path, result)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["config"]["model"], SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER)
            self.assertEqual(payload["config"]["model_spec"], SHOGI_POLICY_VALUE_MODEL_SPEC)
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_token_ids, batch.candidate_move_features, batch.candidate_mask)
            actual = loaded(batch.position_token_ids, batch.candidate_move_features, batch.candidate_mask)

        self.assertTrue(torch.allclose(actual, expected))

    def test_save_and_load_policy_plane_model_preserves_logits(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=2,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                model=SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
            ),
        )
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    ShogiPolicyPlaneValueDataset(examples),
                    batch_size=2,
                    collate_fn=collate_policy_plane_value_samples,
                )
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi-policy-plane.pt"
            save_shogi_policy_value_checkpoint(path, result)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["config"]["model"], SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER)
            self.assertEqual(payload["config"]["model_spec"], SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC)
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_token_ids, batch.legal_action_mask)
            actual = loaded(batch.position_token_ids, batch.legal_action_mask)

        self.assertTrue(torch.allclose(actual, expected))

    def test_load_rejects_missing_model_weights(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
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

    def test_load_rejects_missing_input_schema_id(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
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
            payload["config"].pop("input_schema_id")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "input schema"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_model_spec(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
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
            payload["config"].pop("model_spec")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "model spec"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_model(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
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
            payload["config"].pop("model")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "missing model"):
                load_shogi_policy_value_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
