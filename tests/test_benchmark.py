import unittest

from intrep.benchmark import run_benchmark


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_compares_rule_and_frequency_predictors(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.train_size, 6)
        self.assertEqual(result.test_size, 6)
        self.assertLess(result.rule_accuracy, result.frequency_accuracy)
        self.assertEqual(result.frequency_accuracy, 1.0)

    def test_benchmark_includes_prediction_error_update(self) -> None:
        result = run_benchmark()

        self.assertEqual(result.update_result.prediction_error_type, "unsupported")
        self.assertTrue(result.update_success)
        self.assertEqual(result.update_result.training_size_before, 6)
        self.assertEqual(result.update_result.training_size_after, 7)


if __name__ == "__main__":
    unittest.main()
