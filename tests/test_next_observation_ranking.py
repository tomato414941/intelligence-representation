import unittest

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.next_observation_cases import NextObservationCase
from intrep.next_observation_ranking import (
    evaluate_next_observation_ranking,
    evaluate_next_observation_ranking_summary,
    torch_context_limited_continuation_loss,
)


class NextObservationRankingTest(unittest.TestCase):
    def test_evaluate_next_observation_ranking_scores_against_other_positive_next_values(
        self,
    ) -> None:
        cases = [
            _case("a", "episode a\n", "next a"),
            _case("b", "episode b\n", "next b"),
            _case("c", "episode c\n", "next c"),
        ]
        losses = {
            ("episode a\n", "next a"): 0.1,
            ("episode a\n", "next b"): 0.2,
            ("episode a\n", "next c"): 0.9,
            ("episode b\n", "next a"): 0.2,
            ("episode b\n", "next b"): 0.5,
            ("episode b\n", "next c"): 0.7,
            ("episode c\n", "next a"): 0.4,
            ("episode c\n", "next b"): 0.6,
            ("episode c\n", "next c"): 0.3,
        }
        scored: list[tuple[str, str]] = []

        def score(
            _model: object,
            _tokenizer: ByteTokenizer,
            prefix: str,
            continuation: str,
        ) -> float:
            scored.append((prefix, continuation))
            return losses[(prefix, continuation)]

        metrics = evaluate_next_observation_ranking(
            cases,
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=score,
        )

        self.assertAlmostEqual(metrics.top1_accuracy, 2 / 3)
        self.assertAlmostEqual(metrics.mean_positive_loss, (0.1 + 0.5 + 0.3) / 3)
        self.assertAlmostEqual(metrics.mean_best_distractor_loss, (0.2 + 0.2 + 0.4) / 3)
        self.assertAlmostEqual(metrics.mean_margin, (0.1 - 0.3 + 0.1) / 3)
        self.assertEqual(set(scored), set(losses))

    def test_evaluate_next_observation_ranking_summary_groups_by_modality(
        self,
    ) -> None:
        cases = [
            _case("g1", "grid 1\n", "next g1", modality="grid"),
            _case("g2", "grid 2\n", "next g2", modality="grid"),
            _case("text", "text\n", "next text", modality="language"),
            _case("image", "image\n", "next image", modality="image"),
        ]
        losses = {
            ("grid 1\n", "next g1"): 0.1,
            ("grid 1\n", "next g2"): 0.5,
            ("grid 1\n", "next text"): 0.6,
            ("grid 1\n", "next image"): 0.7,
            ("grid 2\n", "next g1"): 0.3,
            ("grid 2\n", "next g2"): 0.2,
            ("grid 2\n", "next text"): 0.4,
            ("grid 2\n", "next image"): 0.5,
            ("text\n", "next g1"): 0.4,
            ("text\n", "next g2"): 0.9,
            ("text\n", "next text"): 0.8,
            ("text\n", "next image"): 0.7,
            ("image\n", "next g1"): 0.2,
            ("image\n", "next g2"): 0.3,
            ("image\n", "next text"): 0.9,
            ("image\n", "next image"): 0.6,
        }

        def score(
            _model: object,
            _tokenizer: ByteTokenizer,
            prefix: str,
            continuation: str,
        ) -> float:
            return losses[(prefix, continuation)]

        summary = evaluate_next_observation_ranking_summary(
            cases,
            model=object(),
            tokenizer=ByteTokenizer(),
            score_continuation_loss=score,
        )

        self.assertEqual(summary.modality_counts, {"grid": 2, "language": 1, "image": 1})
        self.assertEqual(set(summary.per_modality), {"grid", "language", "image"})
        self.assertAlmostEqual(summary.overall.top1_accuracy, 0.5)
        self.assertAlmostEqual(summary.overall.mean_positive_loss, 0.425)
        self.assertAlmostEqual(summary.overall.mean_best_distractor_loss, 0.35)
        self.assertAlmostEqual(summary.overall.mean_margin, -0.075)

        grid_metrics = summary.per_modality["grid"]
        self.assertAlmostEqual(grid_metrics.top1_accuracy, 1.0)
        self.assertAlmostEqual(grid_metrics.mean_positive_loss, 0.15)
        self.assertAlmostEqual(grid_metrics.mean_best_distractor_loss, 0.4)
        self.assertAlmostEqual(grid_metrics.mean_margin, 0.25)

        language_metrics = summary.per_modality["language"]
        self.assertAlmostEqual(language_metrics.top1_accuracy, 0.0)
        self.assertAlmostEqual(language_metrics.mean_positive_loss, 0.8)
        self.assertAlmostEqual(language_metrics.mean_best_distractor_loss, 0.4)
        self.assertAlmostEqual(language_metrics.mean_margin, -0.4)

    def test_evaluate_next_observation_ranking_requires_distractors(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            evaluate_next_observation_ranking(
                [],
                model=object(),
                tokenizer=ByteTokenizer(),
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            evaluate_next_observation_ranking(
                [_case("only", "only\n", "next")],
                model=object(),
                tokenizer=ByteTokenizer(),
            )

    def test_evaluate_next_observation_ranking_uses_torch_continuation_loss(self) -> None:
        tokenizer = ByteTokenizer()
        model = PrefixSensitiveNextTokenModel(tokenizer)
        cases = [
            _case("room-a", "room A:", "x"),
            _case("room-b", "room B:", "y"),
        ]
        model.prefer_next("room A:", "x")
        model.prefer_next("room B:", "y")

        metrics = evaluate_next_observation_ranking(
            cases,
            model=model,
            tokenizer=tokenizer,
        )

        self.assertEqual(metrics.top1_accuracy, 1.0)
        self.assertLess(metrics.mean_positive_loss, 0.01)
        self.assertGreater(metrics.mean_best_distractor_loss, 10.0)
        self.assertGreater(metrics.mean_margin, 10.0)

    def test_context_limited_loss_keeps_inputs_within_model_context(self) -> None:
        tokenizer = ByteTokenizer()
        model = ContextLengthCheckingModel(vocab_size=tokenizer.vocab_size, context_length=2)

        loss = torch_context_limited_continuation_loss(
            model,
            tokenizer,
            prefix="abcdef",
            continuation="g",
        )

        self.assertGreater(loss, 0.0)
        self.assertTrue(model.observed_lengths)
        self.assertLessEqual(max(model.observed_lengths), 2)


class PrefixSensitiveNextTokenModel(torch.nn.Module):
    def __init__(self, tokenizer: ByteTokenizer) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = tokenizer
        self.preferred_by_prefix: dict[str, int] = {}

    def prefer_next(self, prefix: str, next_token_text: str) -> None:
        token_ids = self.tokenizer.encode(next_token_text)
        if len(token_ids) != 1:
            raise ValueError("test next token text must encode to exactly one byte")
        self.preferred_by_prefix[prefix] = token_ids[0]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (token_ids.size(0), token_ids.size(1), self.tokenizer.vocab_size),
            -10.0,
            device=token_ids.device,
        )
        for batch_index in range(token_ids.size(0)):
            for position in range(token_ids.size(1)):
                context_ids = token_ids[batch_index, : position + 1].tolist()
                context = self.tokenizer.decode(context_ids)
                preferred_token = self.preferred_by_prefix.get(context, 0)
                logits[batch_index, position, preferred_token] = 10.0 + self.bias
        return logits


class _Config:
    def __init__(self, context_length: int) -> None:
        self.context_length = context_length


class ContextLengthCheckingModel(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.config = _Config(context_length)
        self.vocab_size = vocab_size
        self.observed_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.observed_lengths.append(token_ids.size(1))
        if token_ids.size(1) > self.config.context_length:
            raise AssertionError("input exceeded model context length")
        logits = torch.zeros(
            (token_ids.size(0), token_ids.size(1), self.vocab_size),
            device=token_ids.device,
        )
        logits = logits + self.bias
        return logits


def _case(
    case_id: str,
    prefix: str,
    positive_next: str,
    *,
    modality: str = "test",
) -> NextObservationCase:
    return NextObservationCase(
        id=case_id,
        modality=modality,
        prefix=prefix,
        positive_next=positive_next,
    )


if __name__ == "__main__":
    unittest.main()
