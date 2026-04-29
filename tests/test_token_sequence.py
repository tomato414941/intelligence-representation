import unittest

from intrep.token_sequence import TokenSequence, token_sequence_from_ids


class TokenSequenceTest(unittest.TestCase):
    def test_token_sequence_stores_ids_and_optional_loss_mask(self) -> None:
        sequence = token_sequence_from_ids([1, 2, 3], loss_mask=[False, True, True])

        self.assertEqual(sequence.token_ids, (1, 2, 3))
        self.assertEqual(sequence.loss_mask, (False, True, True))

    def test_token_sequence_allows_full_sequence_loss(self) -> None:
        sequence = TokenSequence(token_ids=(1, 2, 3))

        self.assertIsNone(sequence.loss_mask)

    def test_token_sequence_validates_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_ids must not be empty"):
            token_sequence_from_ids([])
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            token_sequence_from_ids([1, -1])
        with self.assertRaisesRegex(ValueError, "loss_mask must match"):
            token_sequence_from_ids([1, 2], loss_mask=[True])
        with self.assertRaisesRegex(ValueError, "at least one training token"):
            token_sequence_from_ids([1, 2], loss_mask=[False, False])


if __name__ == "__main__":
    unittest.main()
