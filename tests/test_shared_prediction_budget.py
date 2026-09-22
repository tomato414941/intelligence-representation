from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction import training
from intrep.problems.shared_prediction.evaluation import evaluate_panel, make_panel
from intrep.problems.shared_prediction.full_evaluation import evaluate_full
from intrep.problems.shared_prediction.sources import build_sources
from tests.test_shared_prediction_replay import assert_state_equal
from tests.test_shared_prediction_sources import make_model, make_recipe, make_tokenizer


def local_base(root):
    from transformers import Lfm2ForCausalLM

    Lfm2ForCausalLM(make_model().core.body.config).save_pretrained(root / "base")
    make_tokenizer().save_pretrained(root / "base")
    return root / "base"


class TrainingBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_requires_an_explicit_positive_finite_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for budget in ({}, {"steps": 0}, {"time_budget_seconds": 0},
                           {"time_budget_seconds": float("nan")}, {"time_budget_seconds": float("inf")},
                           {"training_seconds": -1}):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "budget"):
                    training.train(base=root / "base", root=root, recipe={}, output=root / "output", **budget)

    def test_first_reached_budget_counts_evaluation_and_checkpoint_time(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            base = local_base(root)
            clock = [0.0]
            original_step = JointTrainer.step
            original_evaluate = training.evaluate_panel
            original_save = training.save_checkpoint

            def timed_step(trainer, losses):
                result = original_step(trainer, losses)
                clock[0] += 1
                return result

            def timed_evaluate(*args, **kwargs):
                result = original_evaluate(*args, **kwargs)
                clock[0] += 2
                return result

            def timed_save(*args, **kwargs):
                result = original_save(*args, **kwargs)
                clock[0] += 3
                return result

            options = {"recipe": recipe, "root": root, "prompts": [], "evaluation_examples": 2}
            for limit, steps, updates, elapsed, reason in ((5, 20, 1, 9, "time_budget"),
                                                         (50, 2, 2, 12, "step_budget")):
                clock[0] = 0.0
                output = root / reason
                with self.subTest(reason=reason), \
                        patch.object(training.time, "perf_counter", side_effect=lambda: clock[0]), \
                        patch.object(JointTrainer, "step", timed_step), \
                        patch.object(training, "evaluate_panel", timed_evaluate), \
                        patch.object(training, "save_checkpoint", timed_save):
                    checkpoint = training.train(base=base, output=output, steps=steps,
                                                time_budget_seconds=limit, checkpoint_interval=1, **options)
                report = json.loads((output / "result.json").read_text())
                self.assertEqual(report["completed_steps"], updates)
                self.assertEqual(report["stop_reason"], reason)
                self.assertEqual(report["training_seconds"], updates)
                self.assertEqual(report["elapsed_seconds"], elapsed)
                self.assertEqual(report["timing_seconds"]["checkpoint"], 6)
                self.assertEqual(sum(report["timing_seconds"].values()), elapsed)
                self.assertEqual(report["evaluation_complete"], {"before": True, "after": reason == "step_budget"})
                self.assertEqual(report["comparison_complete"], reason == "step_budget")
                rows = [json.loads(row) for row in (output / "steps.jsonl").read_text().splitlines()]
                self.assertEqual(rows[0]["training_seconds"], 1)
                self.assertEqual(rows[0]["elapsed_seconds"], 3)
                _, _, state = training.load_checkpoint(checkpoint)
                self.assertEqual(state["trainer"]["steps"], updates)

            resumed = training.train(base=None, resume=root / "time_budget/checkpoint.pt",
                                     output=root / "resumed", steps=4, **options)
            straight = training.train(base=base, output=root / "straight", steps=4, **options)
            a, _, state_a = training.load_checkpoint(resumed)
            b, _, state_b = training.load_checkpoint(straight)
            for actual, expected in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for key in ("trainer", "sources", "experience_replay", "torch_rng"):
                assert_state_equal(self, state_a[key], state_b[key])

    def test_setup_can_exhaust_the_budget_and_still_save_a_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            base = local_base(root)
            clock = [0.0]
            original_load = training.load_lfm

            def timed_load(*args, **kwargs):
                model = original_load(*args, **kwargs)
                clock[0] += 2
                return model

            options = {"recipe": recipe, "root": root, "prompts": [], "evaluation_examples": 1}
            with patch.object(training.time, "perf_counter", side_effect=lambda: clock[0]), \
                    patch.object(training, "load_lfm", timed_load):
                initial = training.train(base=base, output=root / "initial", time_budget_seconds=1, **options)
            report = json.loads((root / "initial/result.json").read_text())
            self.assertEqual(report["stop_reason"], "time_budget")
            self.assertEqual(report["completed_steps"], 0)
            self.assertEqual(report["timing_seconds"]["setup"], 2)
            self.assertEqual(report["training_seconds"], 0)
            self.assertEqual(report["evaluation_complete"], {"before": False, "after": False})
            _, _, initial_state = training.load_checkpoint(initial)
            self.assertEqual(initial_state["experience_replay"]["fresh_updates"], {"text_data": 0, "pictures": 0})
            resumed = training.train(base=None, resume=initial, output=root / "resumed", steps=2, **options)
            _, _, state = training.load_checkpoint(resumed)
            self.assertEqual(state["trainer"]["steps"], 2)
            self.assertEqual(sum(state["experience_replay"]["replay_updates"].values()), 1)

    def test_evaluation_interruption_restores_reader_rng_and_model_state(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe = make_recipe(root)
            for scope in ("panel", "full"):
                with self.subTest(scope=scope):
                    model = make_model()
                    sources = build_sources(model, make_tokenizer(), recipe, root)
                    panel = make_panel(sources, 4)
                    states = {name: copy.deepcopy(source.state_dict()) for name, source in sources.items()}
                    rng = torch.get_rng_state().clone()
                    checks = [0]

                    def check_budget():
                        checks[0] += 1
                        if checks[0] == 3:
                            raise training.TimeBudgetExceeded("evaluation budget exhausted")

                    with self.assertRaises(training.TimeBudgetExceeded):
                        if scope == "full":
                            evaluate_full(model, sources, root / scope, generate_answers=False, check_budget=check_budget)
                        else:
                            evaluate_panel(model, sources, panel, generate_answers=False, check_budget=check_budget)
                    self.assertTrue(model.training)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    for name, source in sources.items():
                        assert_state_equal(self, source.state_dict(), states[name])


if __name__ == "__main__":
    unittest.main()
