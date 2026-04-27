import math
import unittest
from dataclasses import dataclass

from intrep.language_modeling_metrics import (
    language_modeling_metrics_from_training_result,
    perplexity_from_loss,
)


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_train_loss: float | None = 2.0
    final_train_loss: float | None = 1.0
    initial_eval_loss: float | None = 3.0
    final_eval_loss: float | None = 2.5


class LanguageModelingMetricsTest(unittest.TestCase):
    def test_perplexity_from_loss(self) -> None:
        self.assertAlmostEqual(perplexity_from_loss(1.0), math.e)
        self.assertIsNone(perplexity_from_loss(None))

    def test_perplexity_from_loss_allows_infinity_for_extreme_loss(self) -> None:
        self.assertEqual(perplexity_from_loss(1000.0), math.inf)

    def test_language_modeling_metrics_from_training_result(self) -> None:
        metrics = language_modeling_metrics_from_training_result(FakeTrainingResult())

        self.assertEqual(metrics["initial_train_loss"], 2.0)
        self.assertEqual(metrics["final_train_loss"], 1.0)
        self.assertEqual(metrics["train_loss_delta"], 1.0)
        self.assertAlmostEqual(metrics["initial_train_perplexity"], math.exp(2.0))
        self.assertAlmostEqual(metrics["final_train_perplexity"], math.exp(1.0))
        self.assertEqual(metrics["eval_loss_delta"], 0.5)
        self.assertAlmostEqual(metrics["initial_eval_perplexity"], math.exp(3.0))
        self.assertAlmostEqual(metrics["final_eval_perplexity"], math.exp(2.5))

    def test_language_modeling_metrics_handles_missing_eval_loss(self) -> None:
        result = FakeTrainingResult(initial_eval_loss=None, final_eval_loss=None)

        metrics = language_modeling_metrics_from_training_result(result)

        self.assertIsNone(metrics["eval_loss_delta"])
        self.assertIsNone(metrics["initial_eval_perplexity"])
        self.assertIsNone(metrics["final_eval_perplexity"])


if __name__ == "__main__":
    unittest.main()
