from __future__ import annotations

import argparse
import copy
import gzip
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.answers import answer_loss
from intrep.problems.shared_prediction.rule_transfer import digit_question, order_question
from intrep.problems.shared_prediction.rule_transfer_data import text_training_examples
from intrep.problems.shared_prediction.rule_transfer_training import (
    LESSON_KEY, RuleLessons, batched_answer_loss, measure_prerequisites, state_digest, validate_training_recipe,
)
from intrep.problems.shared_prediction.sources import Source, build_sources
from intrep.problems.shared_prediction.training import load_checkpoint, save_checkpoint
from tests.test_rule_transfer import OracleFixtureReadout, digit_tokenizer, fixture_panel
from tests.test_shared_prediction_questions import question_recipe
from tests.test_shared_prediction_sources import make_model


def training_fixture(root, *, twelve=False, per_class=4):
    recipe = question_recipe(root)
    if twelve:
        recipe["sources"] = [*[{**recipe["sources"][0], "name": f"text_{index}"} for index in range(11)], recipe["sources"][1]]
    pixels = np.random.default_rng(7).integers(0, 256, size=(10 * per_class, 4, 4), dtype=np.uint8)
    labels = np.repeat(np.arange(10, dtype=np.uint8), per_class)
    with gzip.open(root / "train-images.gz", "wb") as handle:
        handle.write(struct.pack(">IIII", 2051, len(labels), 4, 4) + pixels.tobytes())
    with gzip.open(root / "train-labels.gz", "wb") as handle:
        handle.write(struct.pack(">II", 2049, len(labels)) + labels.tobytes())
    model, tokenizer = make_model(), digit_tokenizer()
    sources = build_sources(model, tokenizer, recipe, root)
    return recipe, model, tokenizer, sources


class StateDigestTests(unittest.TestCase):
    def test_digest_preserves_archived_scalar_and_nested_state_hashes(self):
        cases = [
            ({"empty": [], "values": [None, True, False, 0, -1, 2**70, -0.0, 1.5, 1e-20,
                                      "", "文字", '[;]\\"\n']},
             "410602b2491d238c7ee6e5227c1b1609a68b35f7b8d9a778ad045eafe588db88"),
            ({"nested": [[1, 2], ("a", "b"), [], {"z": [1., 2.], "a": torch.tensor([[1., -2.], [3., 4.]])}],
              "sampler": torch.arange(7)},
             "0d56070646ea47be39ddee186609ca92d7c173fd0f9591127e3a02b0a88e4b38"),
            ({"distinct": [str(i) for i in range(20000)], "float_values": [i / 10 for i in range(100)]},
             "ae12b2ade13dca010b18e90b5d33cee0811a4d030bdd0c769b9e7111ab245108"),
        ]
        for value, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(state_digest(value), expected)
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                state_digest([value])


class LessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batched_loss_matches_existing_loss_and_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, _, sources = training_fixture(Path(directory))
            source = sources["mnist"].reader
            questions = [digit_question(source.read_record(index)) for index in (0, 7, 18)]
            questions.append(order_question(source, [3, 8], modality="text", rule="old", order=list(range(10))))
            model.eval()
            expected = torch.stack([answer_loss(source, row.prompt, row.answer,
                [model.encode(name, *values) for name, values in row.inputs]) for row in questions]).mean()
            expected.backward()
            gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
            model.zero_grad(set_to_none=True)
            actual = batched_answer_loss(source, questions)
            actual.backward()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            for name, parameter in model.named_parameters():
                if name in gradients:
                    torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6)

    def test_counterfactuals_share_sampling_and_only_text_tuition_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, model, _, sources = training_fixture(Path(directory))
            panel, _, _ = fixture_panel()
            original = state_digest({name: source.state_dict() for name, source in sources.items()})
            lanes = []
            for condition in ("a", "b", "control"):
                manifest = {"schema_version": "intrep.rule_transfer_text.v1", "condition": condition,
                            "examples": text_training_examples(panel["orders"], condition)}
                lanes.append(RuleLessons(sources["mnist"].reader, panel["orders"], condition=condition,
                                         batches=(4, 3, 5, 90), manifest=manifest))
            for name in lanes[0].names:
                batches = [lane.questions(name) for lane in lanes]
                self.assertEqual(lanes[0].last_trace[name], lanes[1].last_trace[name])
                self.assertEqual(lanes[0].last_trace[name], lanes[2].last_trace[name])
                for a, b in zip(batches[0], batches[1]):
                    self.assertEqual(a.prompt, b.prompt)
                    for (kind_a, values_a), (kind_b, values_b) in zip(a.inputs, b.inputs):
                        self.assertEqual(kind_a, kind_b)
                        for value_a, value_b in zip(values_a, values_b):
                            torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)
                if name == "text_tuition":
                    self.assertEqual(sum(a.answer != b.answer for a, b in zip(*batches[:2])), 46)
                    self.assertTrue(all(kind == "text" for batch in batches for row in batch for kind, _ in row.inputs))
                else:
                    self.assertEqual([row.answer for row in batches[0]], [row.answer for row in batches[1]])
                    self.assertTrue(all("new order" not in row.prompt for batch in batches for row in batch))
            self.assertEqual(original, state_digest({name: source.state_dict() for name, source in sources.items()}))
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            state = copy.deepcopy(lanes[0].state_dict())
            lanes[0].questions("digit_names")
            expected = lanes[0].last_trace["digit_names"]
            lanes[0].load_state_dict(state)
            lanes[0].questions("digit_names")
            self.assertEqual(expected, lanes[0].last_trace["digit_names"])

    def test_grounding_visits_entire_population_and_refuses_tampered_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, sources = training_fixture(Path(directory))
            panel, _, _ = fixture_panel()
            lesson = RuleLessons(sources["mnist"].reader, panel["orders"], batches=(40, 1, 1, 1))
            lesson.questions("digit_names")
            self.assertEqual(set(lesson.last_trace["digit_names"]), set(range(40)))
            manifest = {"schema_version": "intrep.rule_transfer_text.v1", "condition": "a",
                        "examples": text_training_examples(panel["orders"], "a")}
            manifest["examples"][0]["answer"] = "corrupted"
            with self.assertRaisesRegex(ValueError, "exactly match"):
                RuleLessons(sources["mnist"].reader, panel["orders"], condition="a", manifest=manifest)

    def test_prerequisites_read_only_development_and_never_new_rule_images(self):
        panel, images, labels = fixture_panel()
        source = Source(torch.nn.Linear(1, 1), digit_tokenizer(), {}, Path("."))
        oracle = OracleFixtureReadout(source, panel["orders"]["a"])

        class AuditedOracle:
            def snapshot(self):
                return oracle.snapshot()

            def __call__(self, question, candidates):
                if any(kind == "rgb" for kind, _ in question.inputs):
                    assert "new order" not in question.prompt
                torch.rand(1)
                return oracle(question, candidates)

        state = torch.get_rng_state().clone()
        source.model.train()
        result = measure_prerequisites(source, panel, images, labels, condition="a", readout=AuditedOracle())
        self.assertTrue(result["passed"])
        self.assertEqual(result["new_rule_image_queries"], 0)
        self.assertEqual(oracle.seen_images, {index for row in panel["panels"]["development"] for index in row["indices"]})
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertTrue(source.model.training)

    def test_recipe_preserves_twelve_training_populations_and_reserves_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, _, _, _ = training_fixture(root, twelve=True)
            panel = {"files": {"training_images": {"path": "train-images.gz"}}, "holdout_excluded_indices": [1, 3]}
            reserved = copy.deepcopy(recipe)
            reserved["sources"][-1]["evaluation"]["evaluation_excluded_indices"] = [1, 3]
            validate_training_recipe(reserved, recipe, panel, root)
            with self.assertRaisesRegex(ValueError, "reserved"):
                validate_training_recipe(recipe, recipe, panel, root)
            changed = copy.deepcopy(reserved)
            changed["sources"][0]["records_per_update"] = 8
            with self.assertRaisesRegex(ValueError, "all original"):
                validate_training_recipe(changed, recipe, panel, root)


class TrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_joint_updates_resume_exactly_and_fork_from_common_state(self):
        from scripts.train_rule_transfer import train_trial

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe, model, tokenizer, sources = training_fixture(root, twelve=True)
            panel, images, labels = fixture_panel()
            panel["files"] = {"training_images": {"path": "train-images.gz"}}
            panel["holdout_excluded_indices"] = []
            (root / "panel.json").write_text(json.dumps(panel))
            (root / "recipe.json").write_text(json.dumps(recipe))
            trainer = JointTrainer(model, {name: 1. for name in sources}, learning_rate=.001, optimizer="adamw")
            save_checkpoint(root / "initial/checkpoint.pt", model, trainer, sources, recipe,
                            {name: source.provenance() for name, source in sources.items()}, tokenizer)
            options = dict(initialize=root / "initial/checkpoint.pt", common=None, resume=None, condition="calibration",
                           manifest=None, recipe=root / "recipe.json", panel=root / "panel.json", data_root=root,
                           steps=2, interval=2, threads=1, device="cpu", extension=[], prompts=None,
                           training_seconds=None, stop_when_calibrated=False, stop_when_adapted=False,
                           image_manifest=None, milestones=None,
                           batches=[2, 2, 2, 2], weights=[8., 8., 2., 8.], learning_rate=.001, seed=47)

            def measurement(*args, **kwargs):
                # Synthetic gates exercise checkpoint control flow; no capability claim.
                return {"passed": True, "gates": {}, "condition": kwargs["condition"]}

            with patch("scripts.train_rule_transfer.load_panel", return_value=(panel, images, labels)), \
                 patch("scripts.train_rule_transfer.measure_prerequisites", side_effect=measurement), \
                 patch("scripts.train_rule_transfer.evaluate_panel", return_value={}):
                whole = train_trial(argparse.Namespace(**options, output=root / "whole"))
                first_options = {**options, "steps": 1}
                train_trial(argparse.Namespace(**first_options, output=root / "continued"))
                resumed_options = {**options, "initialize": None, "resume": root / "continued/checkpoint.pt"}
                continued = train_trial(argparse.Namespace(**resumed_options, output=root / "continued"))
                self.assertEqual(whole["final_parameters_sha256"], continued["final_parameters_sha256"])
                _, _, a = load_checkpoint(root / "whole/checkpoint.pt")
                _, _, b = load_checkpoint(root / "continued/checkpoint.pt")
                self.assertEqual(state_digest(a["sources"]), state_digest(b["sources"]))
                self.assertEqual(state_digest(a["trainer"]), state_digest(b["trainer"]))
                self.assertEqual(len(whole["source_progress"]), 12)
                self.assertEqual(whole["parameters"], whole["trainable_parameters"])
                self.assertEqual(len(a["trainer"]["weights"]), 15)
                trials = []
                for condition in ("a", "b"):
                    manifest = {"schema_version": "intrep.rule_transfer_text.v1", "condition": condition,
                                "examples": text_training_examples(panel["orders"], condition)}
                    path = root / f"{condition}.json"
                    path.write_text(json.dumps(manifest))
                    fork_options = {**options, "initialize": None, "common": root / "whole/checkpoint.pt",
                                    "condition": condition, "manifest": path, "steps": 1}
                    trials.append(train_trial(argparse.Namespace(**fork_options, output=root / condition)))
                self.assertEqual(trials[0]["initial_checkpoint_sha256"], trials[1]["initial_checkpoint_sha256"])
                self.assertEqual(trials[0]["initial_parameters_sha256"], whole["final_parameters_sha256"])
                steps = [json.loads((root / condition / "steps.jsonl").read_text()) for condition in ("a", "b")]
                self.assertEqual(steps[0]["background_state_sha256"], steps[1]["background_state_sha256"])
                self.assertEqual(steps[0]["lesson_inputs"], steps[1]["lesson_inputs"])
                self.assertEqual(trials[0]["new_rule_image_training_examples"], 0)
                self.assertIn(LESSON_KEY, a["sources"])


if __name__ == "__main__":
    unittest.main()
