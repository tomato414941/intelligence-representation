import unittest

from intrep.shogi_move_choice import shogi_move_choice_examples_from_usi_moves
from intrep.shogi_move_choice_training import (
    ShogiMoveChoiceTrainingConfig,
    train_shogi_move_choice_model,
    train_shogi_move_choice_model_from_usi_file,
)


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
                num_heads=2,
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
                use_shared_core=False,
            ),
        )

        self.assertLess(result.metrics.final_loss, result.metrics.initial_loss)
        self.assertGreaterEqual(result.metrics.accuracy, 0.5)

    def test_trains_from_usi_file(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "games.txt"
            path.write_text("7g7f 3c3d 2g2f\n", encoding="utf-8")
            result = train_shogi_move_choice_model_from_usi_file(
                str(path),
                config=ShogiMoveChoiceTrainingConfig(
                    max_steps=1,
                    batch_size=2,
                    embedding_dim=8,
                    hidden_dim=16,
                    num_heads=2,
                ),
            )

        self.assertEqual(result.metrics.train_case_count, 3)

    def test_trains_with_eval_split_from_usi_files(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.txt"
            eval_path = Path(directory) / "eval.txt"
            train_path.write_text("7g7f 3c3d 2g2f\n", encoding="utf-8")
            eval_path.write_text("2g2f 8c8d\n", encoding="utf-8")
            result = train_shogi_move_choice_model_from_usi_file(
                str(train_path),
                eval_path=str(eval_path),
                config=ShogiMoveChoiceTrainingConfig(
                    max_steps=1,
                    batch_size=2,
                    embedding_dim=8,
                    hidden_dim=16,
                    num_heads=2,
                ),
            )

        self.assertEqual(result.metrics.train_case_count, 3)
        self.assertEqual(result.metrics.eval_case_count, 2)
        self.assertIsNotNone(result.metrics.eval_loss)
        self.assertIsNotNone(result.metrics.eval_accuracy)


if __name__ == "__main__":
    unittest.main()
