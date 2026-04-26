import unittest

from experiments.predictor_evaluation import (
    PredictionCase,
    evaluate_prediction_case,
    evaluate_prediction_cases,
    smoke_cases,
)
from experiments.predictor_interface import Action, Fact, RuleBasedPredictor


class PredictorEvaluationTest(unittest.TestCase):
    def test_evaluate_prediction_case_marks_correct_prediction(self) -> None:
        case = PredictionCase(
            name="place",
            initial_state=[Fact(subject="佐藤", predicate="has", object="本")],
            action=Action(type="place", actor="佐藤", object="本", target="図書館"),
            expected_fact=Fact(subject="本", predicate="located_at", object="図書館"),
        )

        result = evaluate_prediction_case(case, RuleBasedPredictor())

        self.assertTrue(result.correct)
        self.assertFalse(result.unsupported)
        self.assertEqual(result.predicted_fact.key(), ("本", "located_at", "図書館"))

    def test_expected_unsupported_counts_as_correct(self) -> None:
        case = PredictionCase(
            name="unsupported",
            initial_state=[],
            action=Action(type="throw", actor="佐藤", object="本", target="床"),
            expected_fact=None,
        )

        result = evaluate_prediction_case(case, RuleBasedPredictor())

        self.assertTrue(result.correct)
        self.assertTrue(result.unsupported)

    def test_wrong_prediction_is_incorrect(self) -> None:
        case = PredictionCase(
            name="wrong",
            initial_state=[],
            action=Action(type="place", actor="佐藤", object="本", target="図書館"),
            expected_fact=Fact(subject="本", predicate="located_at", object="机"),
        )

        result = evaluate_prediction_case(case, RuleBasedPredictor())

        self.assertFalse(result.correct)
        self.assertFalse(result.unsupported)

    def test_evaluate_prediction_cases_computes_summary(self) -> None:
        summary = evaluate_prediction_cases(smoke_cases(), RuleBasedPredictor())

        self.assertEqual(len(summary.results), 2)
        self.assertEqual(summary.accuracy, 1.0)
        self.assertEqual(summary.unsupported_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
