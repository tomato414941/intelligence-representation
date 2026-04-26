import unittest

from intrep.predictors import TransformerReadyPredictor
from intrep.transition_data import generate_examples, held_out_object_examples, split_examples


class TransformerReadyPredictorTest(unittest.TestCase):
    def test_predicts_seen_sequence_after_fit(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = TransformerReadyPredictor()
        predictor.fit(train)

        prediction = predictor.predict(train[0].state_before, train[0].action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), train[0].expected_observation.key())

    def test_does_not_generalize_to_held_out_sequence(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = TransformerReadyPredictor()
        predictor.fit(train)
        held_out = held_out_object_examples()[0]

        prediction = predictor.predict(held_out.state_before, held_out.action)

        self.assertIsNone(prediction)


if __name__ == "__main__":
    unittest.main()
