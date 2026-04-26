import unittest

from intrep.sequence import sequence_from_example
from intrep.transition_data import held_out_object_examples, missing_link_examples


class SequenceTest(unittest.TestCase):
    def test_sequence_example_contains_input_tokens_and_target(self) -> None:
        example = held_out_object_examples()[0]

        sequence = sequence_from_example(example)

        self.assertEqual(sequence.id, "unseen_wallet_find")
        self.assertIn("ACTION:find:財布:unknown", sequence.input_tokens)
        self.assertEqual(sequence.input_tokens[-1], "PREDICT")
        self.assertEqual(sequence.target_token, "FACT:財布:located_at:引き出し")

    def test_missing_link_sequence_targets_unsupported(self) -> None:
        example = missing_link_examples()[0]

        sequence = sequence_from_example(example)

        self.assertEqual(sequence.target_token, "UNSUPPORTED")


if __name__ == "__main__":
    unittest.main()
