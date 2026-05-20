import tempfile
import unittest
from pathlib import Path

import shogi
import torch

from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_model_checkpoint
from intrep.problems.shogi_policy_value.inference import ShogiPolicyValueCheckpointEvaluator
from intrep.problems.shogi_policy_value.model import (
    SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID,
    SHOGI_NO_CORE_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model
from intrep.worlds.shogi.policy_plane import shogi_policy_plane_action_index


class ShogiPolicyValueInferenceTest(unittest.TestCase):
    def test_checkpoint_returns_legal_move_priors(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        config = ShogiPolicyValueTrainingConfig(
            core=SHOGI_NO_CORE_MODULE_ID,
            policy_output=SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID,
            embedding_dim=8,
            hidden_dim=16,
        )
        model = build_shogi_policy_value_model(config)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate.pt"
            save_shogi_policy_value_model_checkpoint(checkpoint_path, model, config)
            evaluator = ShogiPolicyValueCheckpointEvaluator.from_checkpoint(checkpoint_path)

        priors, value = evaluator.evaluate_batch(((board.sfen(), legal_moves),))[0]

        self.assertEqual(set(priors), set(legal_moves))
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=6)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_policy_plane_checkpoint_maps_legal_moves_to_action_logits(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        preferred_move = "7g7f"
        action_index = shogi_policy_plane_action_index(preferred_move, turn=board.turn)
        config = ShogiPolicyValueTrainingConfig(
            policy_output=SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
            embedding_dim=8,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
        )
        model = build_shogi_policy_value_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_output.scorer[-1].bias[action_index] = 8.0

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "policy-plane.pt"
            save_shogi_policy_value_model_checkpoint(checkpoint_path, model, config)
            evaluator = ShogiPolicyValueCheckpointEvaluator.from_checkpoint(checkpoint_path)

        priors, value = evaluator.evaluate_batch(((board.sfen(), legal_moves),))[0]

        self.assertEqual(max(priors, key=priors.get), preferred_move)
        self.assertEqual(set(priors), set(legal_moves))
        self.assertAlmostEqual(sum(priors.values()), 1.0, places=6)
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
