from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model
from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_model_checkpoint
from intrep.problems.shogi_policy_value.validate_checkpoint import main
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_MINIMAL_SPLIT_GLOBAL_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
)


class ValidateShogiPolicyValueCheckpointCliTest(unittest.TestCase):
    def test_validates_checkpoint_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "entry"
            config = ShogiPolicyValueTrainingConfig(
                assembly_spec_id=SHOGI_POLICY_VALUE_MINIMAL_SPLIT_GLOBAL_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
                embedding_dim=8,
                hidden_dim=16,
                num_heads=2,
                num_layers=1,
            )
            save_shogi_policy_value_model_checkpoint(checkpoint_path, build_shogi_policy_value_model(config), config)

            with patch(
                "sys.argv",
                [
                    "intrep.problems.shogi_policy_value.validate_checkpoint",
                    str(checkpoint_path),
                ],
            ), patch("sys.stdout", new_callable=StringIO) as stdout:
                main()

            payload = json.loads(stdout.getvalue())

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["checkpoint_path"], str(checkpoint_path))
        self.assertEqual(
            payload["assembly_spec_id"],
            SHOGI_POLICY_VALUE_MINIMAL_SPLIT_GLOBAL_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
        )
        self.assertTrue(payload["checkpoint_id"].startswith("shogi-policy-value:sha256:"))


if __name__ == "__main__":
    unittest.main()
