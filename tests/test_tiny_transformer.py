import unittest

from intrep.tiny_transformer import TinyTransformerConfig, TinyTransformerPredictor
from intrep.torch_sequence import build_vocabulary
from intrep.sequence import sequences_from_examples
from intrep.transition_data import generate_examples, held_out_object_examples, split_examples


class TinyTransformerPredictorTest(unittest.TestCase):
    def test_vocabulary_contains_special_and_training_tokens(self) -> None:
        train, _ = split_examples(generate_examples())

        vocabulary = build_vocabulary(sequences_from_examples(train))

        self.assertIn("<PAD>", vocabulary.token_to_id)
        self.assertIn("<UNK>", vocabulary.token_to_id)
        self.assertIn("PREDICT", vocabulary.token_to_id)

    def test_learns_seen_training_example(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = TinyTransformerPredictor(TinyTransformerConfig(epochs=35, seed=7))
        predictor.fit(train)

        prediction = predictor.predict(train[0].state_before, train[0].action)

        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.key(), train[0].expected_observation.key())

    def test_does_not_claim_held_out_generalization_yet(self) -> None:
        train, _ = split_examples(generate_examples())
        predictor = TinyTransformerPredictor(TinyTransformerConfig(epochs=35, seed=7))
        predictor.fit(train)
        held_out = held_out_object_examples()[0]

        prediction = predictor.predict(held_out.state_before, held_out.action)

        self.assertNotEqual(prediction, held_out.expected_observation)


if __name__ == "__main__":
    unittest.main()
