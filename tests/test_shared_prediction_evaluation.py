from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from intrep.problems.shared_prediction.evaluation import (
    evaluate_panel,
    make_panel,
    paired_comparison,
)
from intrep.problems.shared_prediction.recipe import evaluation_recipe
from intrep.problems.shared_prediction.sources import build_sources
from tests.test_shared_prediction_sources import make_model, make_recipe, make_tokenizer


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import transformers  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("install the lfm extra") from error
        torch.set_num_threads(1)

    def test_fixed_panel_restores_sources_and_measures_identical_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            model = make_model()
            sources = build_sources(model, make_tokenizer(), evaluation_recipe(recipe), root)
            states = copy.deepcopy({name: source.state_dict() for name, source in sources.items()})
            panel = make_panel(sources, 8)
            self.assertEqual(panel, make_panel(sources, 8))
            self.assertEqual(len(panel["pictures"]), 4)
            self.assertEqual(len({row["key"] for row in panel["pictures"]}), 4)
            first = evaluate_panel(model, sources, panel)
            second = evaluate_panel(model, sources, panel)
            self.assertEqual(first, second)
            self.assertEqual(sources["text_data"].state_dict(), states["text_data"])
            self.assertEqual(sources["pictures"].sampler.samples, states["pictures"]["samples"])
            torch.testing.assert_close(sources["pictures"].sampler.order, states["pictures"]["order"])
            self.assertEqual(first["pictures"]["summary"]["accuracy"]["count"], 4)
            comparison = paired_comparison(first, second)
            self.assertEqual(comparison["pictures"]["paired_change"]["loss"]["mean"], 0)
            changed = copy.deepcopy(second)
            changed["pictures"]["rows"].pop()
            with self.assertRaisesRegex(ValueError, "same examples"):
                paired_comparison(first, changed)

    def test_intermediate_evaluation_does_not_change_training_updates(self):
        from transformers import Lfm2ForCausalLM

        from intrep.problems.shared_prediction.training import load_checkpoint, train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = make_recipe(root)
            Lfm2ForCausalLM(make_model().core.body.config).save_pretrained(root / "base")
            make_tokenizer().save_pretrained(root / "base")
            options = {"base": root / "base", "recipe": recipe, "root": root, "steps": 2,
                       "optimizer": "adamw", "evaluation_examples": 4, "prompts": []}
            first = train(**options, output=root / "plain")
            second = train(**options, output=root / "measured", evaluation_interval=1)
            a, _, _ = load_checkpoint(first)
            b, _, _ = load_checkpoint(second)
            for actual, expected in zip(a.parameters(), b.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue((root / "measured/evaluation/step-000001.json").exists())


if __name__ == "__main__":
    unittest.main()
