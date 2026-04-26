import unittest

from intrep.predictors import StateAwarePredictor
from intrep.transition_data import generate_examples, held_out_object_examples, split_examples


class StateAwarePredictorTest(unittest.TestCase):
    def test_predicts_held_out_object_from_state_relations(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = StateAwarePredictor()
        predictor.fit(train)
        case = held_out_object_examples()[0]

        prediction = predictor.predict(case.state_before, case.action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), case.expected_observation.key())

    def test_predicts_all_held_out_object_cases(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = StateAwarePredictor()
        predictor.fit(train)

        predictions = [
            predictor.predict(case.state_before, case.action)
            for case in held_out_object_examples()
        ]

        self.assertTrue(all(prediction is not None for prediction in predictions))


if __name__ == "__main__":
    unittest.main()
