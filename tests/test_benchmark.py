import unittest

from intrep.benchmark import run_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_compares_rule_and_frequency_predictors(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.train_size, 6)
        self.assertEqual(result.test_size, 6)
        self.assertEqual(len(result.frequency_summary.results), 7)
        self.assertLess(result.rule_accuracy, result.frequency_accuracy)
        self.assertGreater(result.frequency_accuracy, 0.8)

    def test_benchmark_includes_prediction_error_update(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.update_result.prediction_error_type, "unsupported")
        self.assertTrue(result.update_success)
        self.assertEqual(result.update_result.training_size_before, 6)
        self.assertEqual(result.update_result.training_size_after, 7)

    def test_benchmark_breaks_out_held_out_object_failure(self) -> None:
        result = run_benchmark()

        seen = result.slice("seen_action_patterns")
        held_out = result.slice("held_out_object")

        self.assertEqual(seen.case_count, 6)
        self.assertEqual(seen.frequency_summary.accuracy, 1.0)
        self.assertEqual(held_out.case_count, 1)
        self.assertEqual(held_out.frequency_summary.accuracy, 0.0)
        self.assertEqual(held_out.frequency_summary.unsupported_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
