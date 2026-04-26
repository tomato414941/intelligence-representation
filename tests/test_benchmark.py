import unittest

from intrep.benchmark import run_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_compares_rule_and_frequency_predictors(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.train_size, 6)
        self.assertEqual(result.test_size, 6)
        self.assertEqual(len(result.frequency_summary.results), 9)
        self.assertLess(result.rule_accuracy, result.frequency_accuracy)
        self.assertLess(result.frequency_accuracy, result.state_aware_accuracy)
        self.assertEqual(result.state_aware_accuracy, 1.0)

    def test_benchmark_includes_prediction_error_update(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.update_result.prediction_error_type, "unsupported")
        self.assertTrue(result.update_success)
        self.assertEqual(result.update_result.training_size_before, 6)
        self.assertEqual(result.update_result.training_size_after, 7)

    def test_benchmark_breaks_out_held_out_object_failure_and_improvement(self) -> None:
        result = run_benchmark()

        seen = result.slice("seen_action_patterns")
        held_out = result.slice("held_out_object")

        self.assertEqual(seen.case_count, 6)
        self.assertEqual(seen.frequency_summary.accuracy, 1.0)
        self.assertEqual(seen.state_aware_summary.accuracy, 1.0)
        self.assertEqual(held_out.case_count, 3)
        self.assertEqual(held_out.frequency_summary.accuracy, 0.0)
        self.assertEqual(held_out.frequency_summary.unsupported_rate, 1.0)
        self.assertEqual(held_out.state_aware_summary.accuracy, 1.0)
        self.assertEqual(held_out.state_aware_summary.unsupported_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
