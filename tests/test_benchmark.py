import unittest

from intrep.benchmark import run_benchmark


class BenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_benchmark()

    def test_benchmark_compares_rule_and_frequency_predictors(self) -> None:
        result = self.result

        self.assertEqual(result.train_size, 6)
        self.assertEqual(result.test_size, 6)
        self.assertEqual(len(result.frequency_summary.results), 15)
        self.assertLess(result.rule_accuracy, result.frequency_accuracy)
        self.assertLess(result.frequency_accuracy, result.state_aware_accuracy)
        self.assertGreater(result.state_aware_accuracy, 0.8)

    def test_benchmark_includes_prediction_error_update(self) -> None:
        result = self.result

        self.assertEqual(result.update_result.prediction_error_type, "unsupported")
        self.assertTrue(result.update_success)
        self.assertEqual(result.update_result.training_size_before, 6)
        self.assertEqual(result.update_result.training_size_after, 7)

    def test_benchmark_breaks_out_held_out_object_failure_and_improvement(self) -> None:
        result = self.result

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

    def test_benchmark_exposes_more_failure_conditions(self) -> None:
        result = self.result

        longer_chain = result.slice("longer_chain")
        missing_link = result.slice("missing_link")
        noisy_distractor = result.slice("noisy_distractor")

        self.assertEqual(longer_chain.case_count, 2)
        self.assertEqual(longer_chain.frequency_summary.unsupported_rate, 1.0)
        self.assertEqual(longer_chain.state_aware_summary.accuracy, 1.0)
        self.assertEqual(longer_chain.state_aware_summary.unsupported_rate, 0.0)

        self.assertEqual(missing_link.case_count, 2)
        self.assertEqual(missing_link.frequency_summary.unsupported_rate, 1.0)
        self.assertEqual(missing_link.state_aware_summary.accuracy, 1.0)
        self.assertEqual(missing_link.state_aware_summary.unsupported_rate, 1.0)

        self.assertEqual(noisy_distractor.case_count, 2)
        self.assertEqual(noisy_distractor.frequency_summary.unsupported_rate, 1.0)
        self.assertEqual(noisy_distractor.state_aware_summary.accuracy, 1.0)
        self.assertEqual(noisy_distractor.state_aware_summary.unsupported_rate, 0.0)

    def test_benchmark_includes_generated_distribution(self) -> None:
        result = self.result

        seen = result.generated_slice("generated_seen")
        held_out_object = result.generated_slice("generated_held_out_object")
        held_out_container = result.generated_slice("generated_held_out_container")
        held_out_location = result.generated_slice("generated_held_out_location")

        self.assertEqual(result.generated_train_size, 12)
        self.assertEqual(seen.frequency_summary.accuracy, 1.0)
        self.assertEqual(held_out_object.frequency_summary.accuracy, 0.0)
        self.assertEqual(held_out_container.frequency_summary.accuracy, 0.5)
        self.assertEqual(held_out_location.frequency_summary.accuracy, 0.0)
        self.assertEqual(held_out_object.state_aware_summary.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
