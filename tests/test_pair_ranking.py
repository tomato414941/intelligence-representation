import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.mixed_corpus import MixedDocument
from intrep.mixed_corpus_evaluation import MixedEnvironmentDocumentPair
from intrep.pair_ranking import (
    evaluate_symbolic_to_natural_ranking,
    torch_next_token_continuation_loss,
    torch_next_token_continuation_losses,
)


class PairRankingTest(unittest.TestCase):
    def test_evaluate_symbolic_to_natural_ranking_scores_against_distractors(self) -> None:
        pairs = [
            _pair("a", "<obs> a", "Natural A."),
            _pair("b", "<obs> b", "Natural B."),
            _pair("c", "<obs> c", "Natural C."),
        ]
        losses = {
            ("<obs> a\n", "Natural A."): 0.2,
            ("<obs> a\n", "Natural B."): 0.8,
            ("<obs> a\n", "Natural C."): 0.6,
            ("<obs> b\n", "Natural A."): 0.4,
            ("<obs> b\n", "Natural B."): 0.5,
            ("<obs> b\n", "Natural C."): 0.7,
            ("<obs> c\n", "Natural A."): 0.9,
            ("<obs> c\n", "Natural B."): 0.3,
            ("<obs> c\n", "Natural C."): 0.4,
        }

        metrics = evaluate_symbolic_to_natural_ranking(
            pairs,
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=lambda _model, _tokenizer, prefix, continuation: losses[
                (prefix, continuation)
            ],
        )

        self.assertAlmostEqual(metrics.top1_accuracy, 1 / 3)
        self.assertAlmostEqual(metrics.mean_correct_loss, (0.2 + 0.5 + 0.4) / 3)
        self.assertAlmostEqual(metrics.mean_best_distractor_loss, (0.6 + 0.4 + 0.3) / 3)
        self.assertAlmostEqual(metrics.mean_margin, (0.4 - 0.1 - 0.1) / 3)

    def test_evaluate_symbolic_to_natural_ranking_requires_distractors(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_symbolic_to_natural_ranking(
                [_pair("a", "<obs> a", "Natural A.")],
                model=object(),
                tokenizer=ByteTokenizer(),
            )

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


def _pair(
    episode_id: str,
    symbolic_content: str,
    natural_content: str,
) -> MixedEnvironmentDocumentPair:
    return MixedEnvironmentDocumentPair(
        episode_id=episode_id,
        symbolic=MixedDocument(
            id=f"env_symbolic_{episode_id}",
            modality="environment_symbolic",
            content=symbolic_content,
        ),
        natural=MixedDocument(
            id=f"env_natural_{episode_id}",
            modality="environment_natural",
            content=natural_content,
        ),
    )


if __name__ == "__main__":
    unittest.main()
