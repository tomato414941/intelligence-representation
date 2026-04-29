import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.pair_ranking import (
    torch_next_token_continuation_loss,
    torch_next_token_continuation_losses,
)


class PairRankingTest(unittest.TestCase):
    def test_torch_next_token_continuation_loss_scores_only_continuation_tokens(self) -> None:
        tokenizer = ByteTokenizer()
        model = FixedLogitModel(vocab_size=tokenizer.vocab_size)
        prefix = "ab"
        continuation = "cd"
        token_ids = tokenizer.encode(prefix + continuation)
        model.prefer_next_tokens(token_ids, wrong_prefix_loss=True)

        loss = torch_next_token_continuation_loss(model, tokenizer, prefix, continuation)

        self.assertLess(loss, 0.01)

    def test_batched_continuation_losses_match_single_loss(self) -> None:
        tokenizer = ByteTokenizer()
        model = UniformLogitModel(vocab_size=tokenizer.vocab_size)
        prefix = "ab"
        continuations = ["cd", "ef"]

        batched_losses = torch_next_token_continuation_losses(
            model,
            tokenizer,
            prefix,
            continuations,
        )
        single_losses = [
            torch_next_token_continuation_loss(model, tokenizer, prefix, continuation)
            for continuation in continuations
        ]

        self.assertEqual(len(batched_losses), 2)
        for batched_loss, single_loss in zip(batched_losses, single_losses, strict=True):
            self.assertAlmostEqual(batched_loss, single_loss, places=5)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.next_tokens: list[int] = []
        self.wrong_prefix_loss = False

    def prefer_next_tokens(self, token_ids: list[int], *, wrong_prefix_loss: bool) -> None:
        self.next_tokens = token_ids[1:]
        self.wrong_prefix_loss = wrong_prefix_loss

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (token_ids.size(0), token_ids.size(1), self.vocab_size),
            -10.0,
            device=token_ids.device,
        )
        for index, next_token in enumerate(self.next_tokens[: token_ids.size(1)]):
            preferred_token = 0 if self.wrong_prefix_loss and index == 0 else next_token
            logits[:, index, preferred_token] = 10.0 + self.bias
        return logits


class UniformLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (token_ids.size(0), token_ids.size(1), self.vocab_size),
            device=token_ids.device,
        ) + self.bias

if __name__ == "__main__":
    unittest.main()
