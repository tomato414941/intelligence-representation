import unittest

import torch

from intrep.token_scoring import next_token_loss, next_token_losses


class TokenScoringTest(unittest.TestCase):
    def test_next_token_loss_scores_only_masked_target_positions(self) -> None:
        logits = torch.full((1, 4, 8), -10.0)
        target_token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        loss_mask = torch.tensor([[False, False, True, True]], dtype=torch.bool)
        logits[0, 0, 0] = 10.0
        logits[0, 1, 3] = 10.0
        logits[0, 2, 4] = 10.0

        loss = next_token_loss(logits, target_token_ids, loss_mask=loss_mask)

        self.assertLess(float(loss.item()), 0.01)

    def test_next_token_loss_rejects_mask_without_predictable_token(self) -> None:
        logits = torch.zeros((1, 2, 8))
        target_token_ids = torch.tensor([[1, 2]], dtype=torch.long)
        loss_mask = torch.tensor([[True, False]], dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "predictable token per row"):
            next_token_loss(logits, target_token_ids, loss_mask=loss_mask)

    def test_next_token_losses_return_one_loss_per_row(self) -> None:
        logits = torch.full((2, 3, 8), -10.0)
        target_token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        loss_mask = torch.tensor([[False, True, False], [False, False, True]], dtype=torch.bool)
        logits[0, 0, 2] = 10.0
        logits[1, 1, 6] = 10.0

        losses = next_token_losses(logits, target_token_ids, loss_mask=loss_mask)

        self.assertEqual(losses.shape, torch.Size([2]))
        self.assertLess(float(losses.max().item()), 0.01)

    def test_next_token_loss_validates_inputs(self) -> None:
        logits = torch.zeros((1, 2, 8))
        target_token_ids = torch.tensor([[1, 2]], dtype=torch.long)
        loss_mask = torch.tensor([[False, True]], dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "logits"):
            next_token_loss(torch.zeros((2, 8)), target_token_ids, loss_mask=loss_mask)
        with self.assertRaisesRegex(ValueError, "dtype torch.long"):
            next_token_loss(logits, target_token_ids.float(), loss_mask=loss_mask)
        with self.assertRaisesRegex(ValueError, "dtype torch.bool"):
            next_token_loss(logits, target_token_ids, loss_mask=loss_mask.long())
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            next_token_loss(logits, torch.tensor([[1, 9]], dtype=torch.long), loss_mask=loss_mask)


if __name__ == "__main__":
    unittest.main()
