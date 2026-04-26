import unittest

from intrep.sequence_predictor import SequenceFeaturePredictor
from intrep.transition_data import generate_examples, held_out_object_examples, missing_link_examples, split_examples


class SequenceFeaturePredictorTest(unittest.TestCase):
    def test_predicts_seen_sequence_after_fit(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = SequenceFeaturePredictor()
        predictor.fit(train)

        prediction = predictor.predict(train[0].state_before, train[0].action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), train[0].expected_observation.key())

    def test_exposes_limit_on_held_out_object(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = SequenceFeaturePredictor()
        predictor.fit(train)
        case = held_out_object_examples()[0]

        prediction = predictor.predict(case.state_before, case.action)

        self.assertNotEqual(prediction, case.expected_observation)

    def test_missing_link_is_not_solved_by_sequence_features(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = SequenceFeaturePredictor()
        predictor.fit(train)
        case = missing_link_examples()[0]

        prediction = predictor.predict(case.state_before, case.action)

        self.assertIsNotNone(prediction)


if __name__ == "__main__":
    unittest.main()
