from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.benchmark_shared_prediction_training import benchmark, scale_recipe, tensor_reference
from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.rule_transfer_data import file_digest
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_KEY, RuleLessons
from intrep.problems.shared_prediction.training import save_checkpoint
from tests.test_rule_transfer import fixture_panel
from tests.test_rule_transfer_training import training_fixture


class TrainingBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import intrep.problems.shared_prediction.record_sources  # noqa: F401
        torch.set_num_threads(1)

    def test_sweep_changes_only_batch_counts_and_does_not_modify_source_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            recipe, _, _, _ = training_fixture(Path(directory), twelve=True)
            original = copy.deepcopy(recipe)
            scaled = scale_recipe(recipe, 4)
            self.assertEqual(recipe, original)
            self.assertEqual(scaled["defaults"], original["defaults"])
            for before, after in zip(original["sources"], scaled["sources"]):
                self.assertEqual({key: value for key, value in before.items() if key != "records_per_update"},
                                 {key: value for key, value in after.items() if key != "records_per_update"})
                self.assertEqual(after["records_per_update"], 8 if after["name"] == "mnist" else 4)
            for multiplier in (0, -1, True, 1.5):
                with self.assertRaises(ValueError):
                    scale_recipe(recipe, multiplier)

    def test_repeats_restore_weights_optimizer_and_every_source_before_timing(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe, model, tokenizer, sources = training_fixture(root, twelve=True)
            panel, _, _ = fixture_panel()
            settings = {"condition": "calibration", "seed": 47, "batches": [2, 2, 2, 2], "learning_rate": .001}
            lessons = RuleLessons(sources["mnist"].reader, panel["orders"], batches=settings["batches"])
            all_sources = {**sources, LESSON_KEY: lessons}
            weights = {name: 1. for name in (*sources, *lessons.names)}
            trainer = JointTrainer(model, weights, learning_rate=.001)
            # Exercise restoration of populated AdamW state and advanced cursors.
            trainer.step({**{name: source.loss for name, source in sources.items()},
                          **{name: lambda name=name: lessons.loss(name) for name in lessons.names}})
            provenance = {name: source.provenance() for name, source in sources.items()}
            provenance[LESSON_KEY] = {"settings": settings, "lessons": lessons.provenance()}
            checkpoint = root / "selected/checkpoint.pt"
            save_checkpoint(checkpoint, model, trainer, all_sources, recipe, provenance, tokenizer,
                            ["intrep.problems.shared_prediction.record_sources"])
            digest = file_digest(checkpoint)
            options = dict(checkpoint=checkpoint, checkpoint_sha256=digest, data_root=root, revision="fixture",
                           steps=3, warmup=1, threads=1, device="cpu", batch_multiplier=1,
                           tensor_reference=root / "reference.pt", write_tensor_reference=True)
            first = benchmark(argparse.Namespace(**options, output=root / "first"))
            options["write_tensor_reference"] = False
            repeated = benchmark(argparse.Namespace(**options, output=root / "repeat"))
            self.assertEqual(first["initial"], repeated["initial"])
            self.assertEqual(first["initial"]["checkpoint_steps"], 1)
            self.assertEqual(file_digest(checkpoint), digest)
            self.assertEqual(repeated["first_update_tensor_comparison"]["gradients"]["relative_l2_error"], 0.)
            traces = [[json.loads(line) for line in (root / name / "steps.jsonl").read_text().splitlines()]
                      for name in ("first", "repeat")]
            for a, b in zip(*traces):
                for key in ("losses", "source_state_sha256", "body", "sources", "lesson_inputs"):
                    self.assertEqual(a[key], b[key])
            options.update(tensor_reference=None, profile=True)
            profiled = benchmark(argparse.Namespace(**options, output=root / "profiled"))
            self.assertTrue(profiled["profiled"])
            self.assertEqual(profiled["initial"], first["initial"])
            profile_rows = [json.loads(line) for line in (root / "profiled/steps.jsonl").read_text().splitlines()]
            for a, b in zip(traces[0], profile_rows):
                for key in ("losses", "source_state_sha256", "body", "sources", "lesson_inputs"):
                    self.assertEqual(a[key], b[key])
            profile = json.loads((root / "profiled/profile-operators.json").read_text())
            operators = {row["name"]: row for row in profile["operators"]}
            self.assertEqual(operators["intrep/update"]["count"], 3)
            self.assertEqual(operators["intrep/optimizer"]["count"], 3)
            self.assertEqual(operators["intrep/source/mnist"]["count"], 3)
            self.assertEqual(operators["intrep/backward/mnist"]["count"], 3)
            self.assertTrue((root / "profiled/profile-trace.json.gz").is_file())
            script = Path(__file__).resolve().parents[1] / "scripts/benchmark_shared_prediction_training.py"
            subprocess.run([sys.executable, str(script), "--checkpoint", str(checkpoint),
                            "--checkpoint-sha256", digest, "--output", str(root / "cli-profile"),
                            "--data-root", str(root), "--revision", "fixture", "--device", "cpu",
                            "--threads", "1", "--warmup", "1", "--steps", "1", "--profile"],
                           cwd=root, check=True, capture_output=True, text=True, timeout=60)
            self.assertTrue(json.loads((root / "cli-profile/result.json").read_text())["profiled"])
            options["profile"] = False
            self.assertEqual(sum(row["measured"] for row in traces[0]), 3)
            self.assertEqual(first["records_per_source"], {name: 6 if name == "mnist" else 3 for name in sources})
            options.update(batch_multiplier=2, tensor_reference=None)
            scaled = benchmark(argparse.Namespace(**options, output=root / "scaled"))
            self.assertEqual(scaled["initial"], first["initial"])
            self.assertEqual(scaled["records_per_source"], {name: count * 2 for name, count in first["records_per_source"].items()})
            self.assertEqual(scaled["lesson_batches"], [4, 4, 4, 4])

    def test_tensor_comparison_rejects_parameter_and_gradient_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(3, 2)
            model(torch.ones(1, 3)).sum().backward()
            path = Path(directory) / "reference.pt"
            tensor_reference(model, path, create=True)
            with torch.no_grad():
                model.weight.add_(1.)
            with self.assertRaisesRegex(ValueError, "parameters differ"):
                tensor_reference(model, path, create=False)
            with torch.no_grad():
                model.weight.sub_(1.)
            model.weight.grad = None
            with self.assertRaisesRegex(ValueError, "different active gradients"):
                tensor_reference(model, path, create=False)


if __name__ == "__main__":
    unittest.main()
