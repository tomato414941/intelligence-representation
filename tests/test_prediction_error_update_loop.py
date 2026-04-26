import unittest

from intrep.transition_data import generate_examples, split_examples
from intrep.update_loop import (
    PredictionErrorUpdateLoop,
    smoke_update_result,
    unseen_wallet_case,
)


class PredictionErrorUpdateLoopTest(unittest.TestCase):
    def test_unknown_case_fails_before_update_and_succeeds_after_update(self) -> None:
        result = smoke_update_result()

        self.assertEqual(result.prediction_error_type, "unsupported")
        self.assertFalse(result.before_correct)
        self.assertTrue(result.after_correct)
        self.assertIsNone(result.predicted_before)
        self.assertEqual(result.predicted_after.key(), ("財布", "located_at", "引き出し"))

    def test_error_case_is_added_to_training_memory(self) -> None:
        result = smoke_update_result()

        self.assertEqual(result.training_size_before, 6)
        self.assertEqual(result.training_size_after, 7)

    def test_no_update_when_prediction_is_already_correct(self) -> None:
        train, _ = split_examples(generate_examples())
        loop = PredictionErrorUpdateLoop(train)
        case = train[0]

        result = loop.update_from_error(case)

        self.assertEqual(result.prediction_error_type, "none")
        self.assertTrue(result.before_correct)
        self.assertTrue(result.after_correct)
        self.assertEqual(result.training_size_before, result.training_size_after)

    def test_loop_keeps_training_examples_after_update(self) -> None:
        train, _ = split_examples(generate_examples())
        loop = PredictionErrorUpdateLoop(train)

        loop.update_from_error(unseen_wallet_case())

        self.assertEqual(len(loop.training_examples), 7)
        self.assertEqual(loop.training_examples[-1].id, "unseen_wallet_find")


if __name__ == "__main__":
    unittest.main()
