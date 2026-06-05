import json
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.problems.shogi_policy_value.samples import (
    ShogiActionPlanePolicyValueDataset,
    ShogiLegalMovePolicyValueDataset,
    collate_legal_move_policy_value_samples,
    collate_action_plane_policy_value_samples,
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
    SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, train_shogi_policy_value_model
from intrep.representation.inputs.shogi_position_module_ids import SHOGI_RICH_POSITION_INPUT_MODULE_ID
from intrep.representation.inputs.shogi_position_features.position_rich import SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH


class ShogiPolicyValueCheckpointTest(unittest.TestCase):
    def test_save_and_load_preserves_logits(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
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
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            self.assertTrue((path / "components" / "input.pt").is_file())
            self.assertTrue((path / "components" / "core.pt").is_file())
            self.assertTrue((path / "components" / "policy_output.pt").is_file())
            self.assertTrue((path / "components" / "value_output.pt").is_file())
            self.assertEqual(payload["components"]["input"]["module_id"], SHOGI_RICH_POSITION_INPUT_MODULE_ID)
            self.assertEqual(payload["components"]["core"]["module_id"], "shared_transformer_core")
            self.assertEqual(payload["components"]["policy_output"]["module_id"], "shogi_legal_move_attention_policy_output")
            self.assertEqual(payload["components"]["value_output"]["module_id"], "scalar_tanh_value_output")
            self.assertEqual(payload["assembly"]["assembly"], SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertEqual(payload["assembly"]["assembly_spec_id"], SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID)
            self.assertNotIn("assembly_spec", payload)
            self.assertNotIn("input_feature_manifest", payload["input"])
            self.assertEqual(payload["input"]["input_feature_manifest_hash"], SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH)
            self.assertTrue(payload["checkpoint"]["checkpoint_id"].startswith(SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX))
            self.assertEqual(len(payload["checkpoint"]["checkpoint_sha256"]), 64)
            self.assertEqual(
                payload["checkpoint"]["checkpoint_id"],
                f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{payload['checkpoint']['checkpoint_sha256']}",
            )
            identity = load_shogi_policy_value_checkpoint_identity(path)
            self.assertEqual(identity.checkpoint_id, payload["checkpoint"]["checkpoint_id"])
            self.assertEqual(identity.checkpoint_sha256, payload["checkpoint"]["checkpoint_sha256"])
            self.assertEqual(identity.assembly, SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertEqual(identity.assembly_spec_id, payload["assembly"]["assembly_spec_id"])
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_features, batch.legal_move_feature_ids, batch.legal_move_mask)
            actual = loaded(batch.position_features, batch.legal_move_feature_ids, batch.legal_move_mask)

        self.assertTrue(torch.allclose(actual, expected))

    def test_save_and_load_action_plane_policy_model_preserves_logits(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(
                max_steps=2,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                assembly_spec_id=SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
            ),
        )
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    ShogiActionPlanePolicyValueDataset(examples),
                    batch_size=2,
                    collate_fn=collate_action_plane_policy_value_samples,
                )
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi-action-plane-policy"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            self.assertEqual(payload["assembly"]["assembly"], SHOGI_POLICY_VALUE_ASSEMBLY_ID)
            self.assertEqual(payload["assembly"]["assembly_spec_id"], SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID)
            self.assertNotIn("assembly_spec", payload)
            loaded = load_shogi_policy_value_checkpoint(path)

        with torch.no_grad():
            expected = result.model(batch.position_features)
            actual = loaded(batch.position_features)

        self.assertTrue(torch.allclose(actual, expected))

    def test_load_rejects_missing_model_weights(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            (path / "components" / "core.pt").unlink()

            with self.assertRaisesRegex(ValueError, "component file"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_checkpoint_identity(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["checkpoint"].pop("checkpoint_id")
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_checkpoint_identity(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["checkpoint"]["checkpoint_sha256"] = "0" * 64
            payload["checkpoint"]["checkpoint_id"] = f"{SHOGI_POLICY_VALUE_CHECKPOINT_ID_PREFIX}{'0' * 64}"
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "checkpoint identity"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_input_schema_id(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["input"].pop("input_schema_id")
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "input schema"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_input_feature_manifest_hash(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["input"].pop("input_feature_manifest_hash")
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "input feature manifest"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_input_feature_manifest_hash(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["input"]["input_feature_manifest_hash"] = "0" * 64
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "input feature manifest"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_assembly_spec(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["assembly"].pop("assembly_spec_id")
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "assembly_spec_id"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_missing_assembly(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["assembly"].pop("assembly")
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "assembly"):
                load_shogi_policy_value_checkpoint(path)

    def test_load_rejects_changed_assembly_spec_id(self) -> None:
        examples = shogi_move_policy_value_examples_from_test_moves(("7g7f", "3c3d"))
        result = train_shogi_policy_value_model(
            examples,
            config=ShogiPolicyValueTrainingConfig(assembly_spec_id=SHOGI_POLICY_VALUE_RICH_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shogi"
            save_shogi_policy_value_checkpoint(path, result)
            payload = _load_manifest(path)
            payload["assembly"]["assembly_spec_id"] = SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID
            _write_manifest(path, payload)

            with self.assertRaisesRegex(ValueError, "component module"):
                load_shogi_policy_value_checkpoint(path)


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("manifest must be an object")
    return payload


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
