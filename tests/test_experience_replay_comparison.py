from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction import training
from scripts.run_experience_replay_comparison import compare_conditions, run_comparison
from tests.test_shared_prediction_budget import local_base
from tests.test_shared_prediction_sources import make_recipe


class ReplayComparisonTests(unittest.TestCase):
    def comparison(self, root, *, final_evaluation_delay=0, full=False, time_budget_seconds=80):
        clock = [0.0]
        evaluations = [0]
        original_load = training.load_lfm
        original_step = JointTrainer.step
        original_evaluate = training.evaluate_full if full else training.evaluate_panel
        original_save = training.save_checkpoint

        def load(*args, **kwargs):
            result = original_load(*args, **kwargs)
            clock[0] += 3
            return result

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            clock[0] += 1
            return result

        def evaluate(*args, **kwargs):
            result = original_evaluate(*args, **kwargs)
            evaluations[0] += 1
            clock[0] += 2 + (final_evaluation_delay if evaluations[0] == 4 else 0)
            return result

        def save(*args, **kwargs):
            result = original_save(*args, **kwargs)
            clock[0] += 3
            return result

        recipe = make_recipe(root)
        base = local_base(root)
        with patch.object(training.time, "perf_counter", side_effect=lambda: clock[0]), \
                patch.object(training, "load_lfm", load), patch.object(JointTrainer, "step", step), \
                patch.object(training, "evaluate_full" if full else "evaluate_panel", evaluate), \
                patch.object(training, "save_checkpoint", save):
            return run_comparison(base=base, recipe=recipe, root=root, output=root / "comparison",
                                  device="cpu", threads=1, time_budget_seconds=time_budget_seconds,
                                  prompts=[], evaluation_examples=None if full else 2)

    def test_real_training_shares_initial_state_and_budgets_and_pairs_both_evaluation_modes(self):
        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as directory, \
                    contextlib.redirect_stdout(io.StringIO()):
                root = Path(directory)
                result = self.comparison(root, full=full)
                output = root / "comparison"
                plan = json.loads((output / "plan.json").read_text())
                self.assertTrue(result["complete"])
                self.assertEqual(result["worker_elapsed_seconds"], 74)
                self.assertEqual(plan["calibration_elapsed_seconds"], 12)
                self.assertEqual(plan["condition_time_budget_seconds"], 33.5)
                self.assertEqual(plan["condition_training_seconds"], 20.5)
                self.assertEqual(plan["evaluation_examples"], None if full else 2)
                self.assertTrue(result["matched_recipe_provenance_evaluation_and_initial_results"])
                self.assertTrue(result["matched_time_budgets_and_device"])
                self.assertEqual(set(result["sources"]), {"text_data", "pictures"})
                for name, row in result["conditions"].items():
                    self.assertEqual(row["completed_steps"], 21)
                    self.assertEqual(row["training_seconds"], 21)
                    self.assertEqual(row["elapsed_seconds"], 31)
                    self.assertEqual(sum(row["timing_seconds"].values()), 31)
                    _, _, payload = training.load_checkpoint(output / name / "checkpoint.pt")
                    self.assertEqual(payload["trainer"]["steps"], 21)
                self.assertEqual(sum(result["conditions"]["no_replay"]["experience_replay"]["replay_updates"].values()), 0)
                self.assertEqual(sum(result["conditions"]["replay_1to1"]["experience_replay"]["replay_updates"].values()), 10)
                for source, comparison in result["sources"].items():
                    expected = (result["conditions"]["replay_1to1"]["evaluation_after"][source]
                                - result["conditions"]["no_replay"]["evaluation_after"][source])
                    self.assertAlmostEqual(comparison["paired_change"]["loss"]["mean"], expected)
                with self.assertRaises(FileExistsError):
                    run_comparison(base=root / "base", recipe=make_recipe(root), root=root, output=output,
                                   device="cpu", time_budget_seconds=80)
                (output / "no_replay/provenance.json").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "provenance"):
                    compare_conditions(output, check_budget=lambda: None)

    def test_insufficient_budget_stops_after_calibration_without_starting_an_unmatched_condition(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "remaining budget"):
                self.comparison(root, time_budget_seconds=45)
            output = root / "comparison"
            result = json.loads((output / "comparison.json").read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["worker_elapsed_seconds"], 12)
            self.assertTrue((output / "calibration/checkpoint.pt").exists())
            self.assertFalse((output / "no_replay").exists())
            self.assertFalse((output / "replay_1to1").exists())

    def test_unfinished_final_evaluation_preserves_checkpoint_without_claiming_comparison(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "complete before/after"):
                self.comparison(root, final_evaluation_delay=40)
            output = root / "comparison"
            result = json.loads((output / "comparison.json").read_text())
            plan = json.loads((output / "plan.json").read_text())
            self.assertFalse(result["complete"])
            self.assertNotIn("sources", result)
            self.assertTrue((output / plan["condition_order"][0] / "checkpoint.pt").exists())
            self.assertFalse((output / plan["condition_order"][1]).exists())

    def test_budget_is_required_finite_and_positive_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in (0, -1, float("nan"), float("inf"), True):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "budget"):
                    run_comparison(base=root, recipe={}, root=root, output=root / "output",
                                   device="cpu", time_budget_seconds=value)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
