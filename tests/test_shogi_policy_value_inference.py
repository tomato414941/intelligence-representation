import tempfile
import unittest
from pathlib import Path

import shogi
import torch

from intrep.problems.shogi_policy_value.checkpoint import save_shogi_policy_value_model_checkpoint
from intrep.problems.shogi_policy_value.inference import ShogiPolicyValueCheckpointEvaluator
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
)
from intrep.problems.shogi_policy_value.training import ShogiPolicyValueTrainingConfig, build_shogi_policy_value_model
from intrep.representation.outputs.shogi_action_plane_policy_encoding import shogi_action_plane_policy_action_index


class ShogiPolicyValueInferenceTest(unittest.TestCase):
    def test_checkpoint_returns_legal_move_priors(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        config = ShogiPolicyValueTrainingConfig(
            assembly_spec_id=SHOGI_POLICY_VALUE_RICH_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
            embedding_dim=8,
            hidden_dim=16,
        )
        model = build_shogi_policy_value_model(config)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "candidate"
            save_shogi_policy_value_model_checkpoint(checkpoint_path, model, config)
            evaluator = ShogiPolicyValueCheckpointEvaluator.from_checkpoint(checkpoint_path)

        priors, value = evaluator.evaluate_batch(((board.sfen(), legal_moves),))[0]

        self.assertEqual(len(priors), len(legal_moves))
        self.assertAlmostEqual(sum(priors), 1.0, places=6)
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)

    def test_action_plane_policy_checkpoint_maps_legal_moves_to_action_logits(self) -> None:
        board = shogi.Board()
        legal_moves = tuple(sorted(move.usi() for move in board.legal_moves))
        preferred_move = "7g7f"
        action_index = shogi_action_plane_policy_action_index(preferred_move, turn=board.turn)
        config = ShogiPolicyValueTrainingConfig(
            assembly_spec_id=SHOGI_POLICY_VALUE_RICH_ACTION_PLANE_POLICY_ASSEMBLY_SPEC_ID,
            embedding_dim=8,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
        )
        model = build_shogi_policy_value_model(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.policy_output.action_logit_bias[action_index] = 8.0

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "action-plane-policy"
            save_shogi_policy_value_model_checkpoint(checkpoint_path, model, config)
            evaluator = ShogiPolicyValueCheckpointEvaluator.from_checkpoint(checkpoint_path)

        priors, value = evaluator.evaluate_batch(((board.sfen(), legal_moves),))[0]

        self.assertTrue(evaluator.accepts_action_indices)
        self.assertEqual(legal_moves[max(range(len(priors)), key=lambda index: priors[index])], preferred_move)
        self.assertEqual(len(priors), len(legal_moves))
        self.assertAlmostEqual(sum(priors), 1.0, places=6)
        self.assertEqual(value, 0.0)

        action_indices = tuple(shogi_action_plane_policy_action_index(move, turn=board.turn) for move in legal_moves)
        priors_from_indices, value = evaluator.evaluate_batch(((board.sfen(), (), action_indices),))[0]

        self.assertEqual(len(priors_from_indices), len(action_indices))
        self.assertEqual(
            max(range(len(priors_from_indices)), key=lambda index: priors_from_indices[index]),
            legal_moves.index(preferred_move),
        )
        self.assertAlmostEqual(sum(priors_from_indices), 1.0, places=6)
        self.assertEqual(value, 0.0)


if __name__ == "__main__":
    unittest.main()
