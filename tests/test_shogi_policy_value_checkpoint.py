import tempfile
import unittest
from pathlib import Path

import torch

from intrep.problems.shogi_policy_value.samples import (
    ShogiPolicyPlaneValueDataset,
    ShogiLegalMovePolicyValueDataset,
    collate_legal_move_policy_value_samples,
    collate_policy_plane_value_samples,
)
from tests.shogi_test_helpers import shogi_move_policy_value_examples_from_test_moves
from intrep.problems.shogi_policy_value.checkpoint import (
    SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX,
    load_shogi_policy_value_checkpoint,
    load_shogi_policy_value_checkpoint_identity,
    save_shogi_policy_value_checkpoint,
)
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_ASSEMBLY_ID,
    SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID,
    shogi_policy_value_assembly_spec_for_id,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, train_shogi_policy_value_model
from intrep.representation.inputs.shogi_position_features.position_rich import SHOGI_RICH_POSITION_FEATURE_MANIFEST, SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH


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
                    ShogiLegalMovePolicyValueDataset(examples),
                    batch_size=2,
                    collate_fn=collate_legal_move_policy_value_samples,
                )
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi.pt"
            save_shogi_policy_value_checkpoint(path, result)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["config"]["assembly"], SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertNotIn("input", payload["config"])
            self.assertNotIn("core", payload["config"])
            self.assertNotIn("policy_output", payload["config"])
            self.assertNotIn("value_output", payload["config"])
            self.assertEqual(payload["config"]["assembly_spec_id"], payload["config"]["assembly_spec"]["assembly_spec_id"])
            self.assertEqual(
                payload["config"]["assembly_spec"],
                shogi_policy_value_assembly_spec_for_id(SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID),
            )
            self.assertEqual(payload["config"]["input_feature_manifest"], SHOGI_RICH_POSITION_FEATURE_MANIFEST)
            self.assertEqual(payload["config"]["input_feature_manifest_hash"], SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH)
            self.assertTrue(payload["config"]["checkpoint_id"].startswith(SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX))
            self.assertEqual(len(payload["config"]["checkpoint_sha256"]), 64)
            self.assertEqual(
                payload["config"]["checkpoint_id"],
                f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{payload['config']['checkpoint_sha256']}",
            )
            identity = load_shogi_policy_value_checkpoint_identity(path)
            self.assertEqual(identity.checkpoint_id, payload["config"]["checkpoint_id"])
            self.assertEqual(identity.checkpoint_sha256, payload["config"]["checkpoint_sha256"])
            self.assertEqual(identity.assembly, SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertEqual(identity.assembly_spec_id, payload["config"]["assembly_spec_id"])
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_features, batch.legal_move_features, batch.legal_move_mask)
            actual = loaded(batch.position_features, batch.legal_move_features, batch.legal_move_mask)

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
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID,
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
            self.assertEqual(payload["config"]["assembly"], SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertNotIn("input", payload["config"])
            self.assertNotIn("core", payload["config"])
            self.assertNotIn("policy_output", payload["config"])
            self.assertNotIn("value_output", payload["config"])
            self.assertEqual(payload["config"]["assembly_spec_id"], payload["config"]["assembly_spec"]["assembly_spec_id"])
            self.assertEqual(
                payload["config"]["assembly_spec"],
                shogi_policy_value_assembly_spec_for_id(SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID),
            )
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_features, batch.legal_action_mask)
            actual = loaded(batch.position_features, batch.legal_action_mask)

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

            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_checkpoint_identity(self) -> None:
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
            payload["config"].pop("checkpoint_id")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_checkpoint_identity(self) -> None:
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
            payload["config"]["checkpoint_sha256"] = "0" * 64
            payload["config"]["checkpoint_id"] = f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{'0' * 64}"
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
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

    def test_load_rejects_missing_input_feature_manifest_hash(self) -> None:
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
            payload["config"].pop("input_feature_manifest_hash")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "input feature manifest"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_input_feature_manifest(self) -> None:
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
            payload["config"]["input_feature_manifest"] = {
                **payload["config"]["input_feature_manifest"],
                "square_feature_count": 999,
            }
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "input feature manifest"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_assembly_spec(self) -> None:
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
            payload["config"].pop("assembly_spec")
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "assembly spec"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_assembly_spec_id(self) -> None:
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
            payload["config"]["assembly_spec_id"] = SHOGI_POLICY_VALUE_RICH_POLICY_PLANE_ASSEMBLY_SPEC_ID
            torch.save(payload, path)

            with self.assertRaisesRegex(ValueError, "assembly spec"):
                load_shogi_policy_value_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
