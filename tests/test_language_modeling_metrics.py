import math
import unittest
from dataclasses import dataclass

from intrep.language_modeling_metrics import (
    language_modeling_metrics_from_training_result,
    perplexity_from_loss,
)


@dataclass(frozen=True)
class FakeTrainingResult:
    initial_loss: float = 4.0
    final_loss: float = 2.5
    best_loss: float = 2.25
    loss_reduction: float = 1.5
    loss_reduction_ratio: float = 0.375
    initial_train_loss: float | None = 2.0
    final_train_loss: float | None = 1.0
    initial_eval_loss: float | None = 3.0
    final_eval_loss: float | None = 2.5
    eval_split: str = "held_out"
    generalization_eval: bool = True
    warnings: tuple[str, ...] = ()


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
        self.assertEqual(metrics["initial_step_loss"], 4.0)
        self.assertEqual(metrics["final_step_loss"], 2.5)
        self.assertEqual(metrics["best_step_loss"], 2.25)
        self.assertEqual(metrics["step_loss_delta"], 1.5)
        self.assertEqual(metrics["step_loss_delta_ratio"], 0.375)
        self.assertEqual(metrics["train_loss_delta"], 1.0)
        self.assertAlmostEqual(metrics["initial_train_perplexity"], math.exp(2.0))
        self.assertAlmostEqual(metrics["final_train_perplexity"], math.exp(1.0))
        self.assertEqual(metrics["eval_loss_delta"], 0.5)
        self.assertAlmostEqual(metrics["initial_eval_perplexity"], math.exp(3.0))
        self.assertAlmostEqual(metrics["final_eval_perplexity"], math.exp(2.5))
        self.assertEqual(metrics["eval_split"], "held_out")
        self.assertTrue(metrics["generalization_eval"])
        self.assertEqual(metrics["warnings"], [])

    def test_language_modeling_metrics_handles_missing_eval_loss(self) -> None:
        result = FakeTrainingResult(initial_eval_loss=None, final_eval_loss=None)

        metrics = language_modeling_metrics_from_training_result(result)

        self.assertIsNone(metrics["eval_loss_delta"])
        self.assertIsNone(metrics["initial_eval_perplexity"])
        self.assertIsNone(metrics["final_eval_perplexity"])


if __name__ == "__main__":
    unittest.main()
