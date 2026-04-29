import unittest

import torch

from intrep.token_sequence import (
    HiddenSequence,
    TokenSequence,
    hidden_sequence_from_embeddings,
    token_sequence_from_ids,
)


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

    def test_hidden_sequence_stores_embeddings_and_optional_loss_mask(self) -> None:
        embeddings = torch.zeros((3, 4), dtype=torch.float32)

        sequence = hidden_sequence_from_embeddings(embeddings, loss_mask=[False, True, True])

        self.assertIs(sequence.embeddings, embeddings)
        self.assertEqual(sequence.loss_mask, (False, True, True))

    def test_hidden_sequence_allows_full_sequence_loss(self) -> None:
        sequence = HiddenSequence(embeddings=torch.zeros((3, 4), dtype=torch.float32))

        self.assertIsNone(sequence.loss_mask)

    def test_hidden_sequence_validates_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            hidden_sequence_from_embeddings(torch.zeros((1, 2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "sequence length"):
            hidden_sequence_from_embeddings(torch.zeros((0, 4), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "hidden size"):
            hidden_sequence_from_embeddings(torch.zeros((3, 0), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "floating point"):
            hidden_sequence_from_embeddings(torch.zeros((3, 4), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "loss_mask must match"):
            hidden_sequence_from_embeddings(torch.zeros((3, 4), dtype=torch.float32), loss_mask=[True])
        with self.assertRaisesRegex(ValueError, "at least one training position"):
            hidden_sequence_from_embeddings(
                torch.zeros((3, 4), dtype=torch.float32),
                loss_mask=[False, False, False],
            )


if __name__ == "__main__":
    unittest.main()
