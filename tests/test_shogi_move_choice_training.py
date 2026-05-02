import unittest

from intrep.shogi_move_choice import shogi_move_choice_examples_from_usi_moves
from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingConfig, train_shogi_move_choice_model


class ShogiMoveChoiceTrainingTest(unittest.TestCase):
    def test_trains_for_one_step(self) -> None:
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d", "2g2f"))

        result = train_shogi_move_choice_model(
            examples,
            config=ShogiMoveChoiceTrainingConfig(
                max_steps=1,
                batch_size=2,
                embedding_dim=8,
                hidden_dim=16,
            ),
        )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertGreater(result.metrics.initial_loss, 0.0)
        self.assertGreater(result.metrics.final_loss, 0.0)

    def test_can_overfit_tiny_move_sequence(self) -> None:
        examples = shogi_move_choice_examples_from_usi_moves(("7g7f", "3c3d"))

        result = train_shogi_move_choice_model(
            examples,
            config=ShogiMoveChoiceTrainingConfig(
                max_steps=80,
                batch_size=2,
                learning_rate=0.02,
                embedding_dim=8,
                hidden_dim=16,
            ),
        )

        self.assertLess(result.metrics.final_loss, result.metrics.initial_loss)
        self.assertGreaterEqual(result.metrics.accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
