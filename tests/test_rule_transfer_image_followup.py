from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.rule_transfer import image_record
from intrep.problems.shared_prediction.rule_transfer_data import (
    file_digest, image_training_examples, text_training_examples, validate_image_manifest,
)
from intrep.problems.shared_prediction.rule_transfer_training import (
    LESSON_KEY, RuleLessons, measure_image_followup, state_digest,
)
from intrep.problems.shared_prediction.sources import Source
from intrep.problems.shared_prediction.training import load_checkpoint, save_checkpoint
from tests.test_rule_transfer import OracleFixtureReadout, digit_tokenizer, fixture_panel
from tests.test_rule_transfer_training import training_fixture


def image_manifest(images, labels, panel, *, count=32):
    return {"schema_version": "intrep.rule_transfer_images.v1", "source_split": "train",
            "seed": 71, "order": panel["orders"]["a"],
            "examples": image_training_examples(images, labels, panel["orders"], count=count)}


class ImageLessonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_nested_support_is_balanced_unique_and_identifies_the_whole_order(self):
        panel, images, labels = fixture_panel()
        all_examples = image_training_examples(images, labels, panel["orders"])
        for count in (32, 128, 512):
            manifest = image_manifest(images, labels, panel, count=count)
            selected = manifest["examples"]
            self.assertEqual(selected, all_examples[:count])
            self.assertEqual(sum(row["answer"] == "yes" for row in selected), count // 2)
            self.assertEqual(len({index for row in selected for index in row["indices"]}), count)
            self.assertEqual(len({value for row in selected for value in row["image_sha256"]}), count)
            positive = {tuple(row["digits"]) for row in selected if row["answer"] == "yes"}
            self.assertTrue(set(zip(panel["orders"]["a"], panel["orders"]["a"][1:])) <= positive)
            if count >= 128:
                self.assertEqual(len(positive), 45)
            validate_image_manifest(manifest, images, labels, panel["orders"])
        for field, value in (("answer", "corrupt"), ("indices", [0, 1]), ("image_sha256", ["x", "y"])):
            corrupted = image_manifest(images, labels, panel)
            corrupted["examples"][0][field] = value
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_image_manifest(corrupted, images, labels, panel["orders"])
        with self.assertRaisesRegex(ValueError, "distinct"):
            image_training_examples(np.zeros_like(images), labels, panel["orders"])

    def test_a_and_control_receive_identical_images_without_new_text_tuition(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, sources = training_fixture(Path(directory), per_class=16)
            panel, _, _ = fixture_panel()
            reader = sources["mnist"].reader
            manifest = image_manifest(reader.images, reader.labels, panel)
            lanes = [RuleLessons(reader, panel["orders"], condition=name, image_manifest=manifest)
                     for name in ("a", "control")]
            for name in lanes[0].names:
                questions = [lane.questions(name) for lane in lanes]
                self.assertEqual(lanes[0].last_trace, lanes[1].last_trace)
                for a, b in zip(*questions):
                    self.assertEqual((a.prompt, a.answer), (b.prompt, b.answer))
                    for (kind_a, values_a), (kind_b, values_b) in zip(a.inputs, b.inputs):
                        self.assertEqual(kind_a, kind_b)
                        for value_a, value_b in zip(values_a, values_b):
                            torch.testing.assert_close(value_a, value_b, rtol=0, atol=0)
                    if "new order" in a.prompt:
                        self.assertEqual([kind for kind, _ in a.inputs], ["text", "rgb", "text", "rgb"])
                        expected_markers = [reader.ids(reader.text_ids(f"\nObservation {index}:\n")) for index in (1, 2)]
                        for values, marker in zip([values for kind, values in a.inputs if kind == "text"], expected_markers):
                            torch.testing.assert_close(values[0], marker, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "not active"):
                lanes[0].questions("text_tuition")
            saved = copy.deepcopy(lanes[0].state_dict())
            lanes[0].questions("image_tuition")
            expected = copy.deepcopy(lanes[0].last_trace)
            lanes[0].load_state_dict(saved)
            lanes[0].questions("image_tuition")
            self.assertEqual(expected, lanes[0].last_trace)

    def test_image_measurement_uses_only_development_and_support_and_restores_rng(self):
        panel, images, labels = fixture_panel()
        source = Source(torch.nn.Linear(1, 1), digit_tokenizer(), {}, Path("."))
        training_images = images.copy()
        for index in range(len(training_images)):
            training_images[index, 1, :2] = (index + 2000) % 256, (index + 2000) // 256
        manifest = image_manifest(training_images, labels, panel)
        lessons = SimpleNamespace(image_manifest=manifest,
            source=SimpleNamespace(read_record=lambda index: image_record(source, training_images[index])))
        oracle = OracleFixtureReadout(source, panel["orders"]["a"])

        class RandomizedOracle:
            def snapshot(self):
                return oracle.snapshot()

            def __call__(self, question, candidates):
                torch.rand(1)
                return oracle(question, candidates)

        source.model.train()
        state = torch.get_rng_state().clone()
        result = measure_image_followup(source, panel, images, labels, lessons,
                                        condition="a", readout=RandomizedOracle())
        self.assertTrue(result["passed"])
        self.assertEqual(result["support_accuracy"], 1)
        expected_images = {index for row in panel["panels"]["development"] for index in row["indices"]}
        expected_images.update(index + 2000 for row in manifest["examples"] for index in row["indices"])
        self.assertEqual(oracle.seen_images, expected_images)
        self.assertEqual(result["new_rule_image_queries"], 90 + 32)
        self.assertEqual(result["development_new_rule_image_queries"], 90)
        self.assertEqual(result["support_new_rule_image_queries"], 32)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertTrue(source.model.training)

    def test_image_forks_reset_tuition_and_optimizer_then_resume_all_twelve_sources_exactly(self):
        from scripts.train_rule_transfer import train_trial

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe, model, tokenizer, sources = training_fixture(root, twelve=True, per_class=16)
            panel, images, labels = fixture_panel()
            panel.update(files={"training_images": {"path": "train-images.gz"}}, holdout_excluded_indices=[])
            (root / "panel.json").write_text(json.dumps(panel))
            (root / "recipe.json").write_text(json.dumps(recipe))
            trainer = JointTrainer(model, {name: 1. for name in sources}, learning_rate=.001, optimizer="adamw")
            save_checkpoint(root / "initial/checkpoint.pt", model, trainer, sources, recipe,
                            {name: source.provenance() for name, source in sources.items()}, tokenizer)
            options = dict(initialize=root / "initial/checkpoint.pt", common=None, resume=None, condition="calibration",
                           manifest=None, image_manifest=None, recipe=root / "recipe.json", panel=root / "panel.json",
                           data_root=root, steps=1, interval=1, milestones=None, threads=1, device="cpu", extension=[],
                           prompts=None, training_seconds=None, stop_when_calibrated=False, stop_when_adapted=False,
                           batches=[2, 2, 2, 2], weights=[8., 8., 2., 8.], learning_rate=.001, seed=47)

            def measure(*args, **kwargs):
                # Synthetic gates test control flow and state restoration, not model competence.
                return {"passed": True, "gates": {}, "condition": kwargs["condition"]}

            def image_measure(*args, **kwargs):
                return {**measure(*args, **kwargs), "passed": False, "split": "development",
                        "new_rule_image_queries": 122, "support_accuracy": .5}

            manifest = image_manifest(sources["mnist"].reader.images, sources["mnist"].reader.labels, panel)
            manifest.update(panel_sha256=file_digest(root / "panel.json"), files={
                name: {"path": "train-" + name + ".gz", "sha256": file_digest(root / ("train-" + name + ".gz"))}
                for name in ("images", "labels")})
            image_path = root / "image.json"
            image_path.write_text(json.dumps(manifest))
            with patch("scripts.train_rule_transfer.load_panel", return_value=(panel, images, labels)), \
                 patch("scripts.train_rule_transfer.measure_prerequisites", side_effect=measure), \
                 patch("scripts.train_rule_transfer.measure_image_followup", side_effect=image_measure), \
                 patch("scripts.train_rule_transfer.evaluate_panel", return_value={}):
                train_trial(argparse.Namespace(**options, output=root / "common"))
                parents = {}
                for name in ("a", "control"):
                    path = root / (name + ".json")
                    path.write_text(json.dumps({"schema_version": "intrep.rule_transfer_text.v1", "condition": name,
                                               "examples": text_training_examples(panel["orders"], name)}))
                    fork = {**options, "initialize": None, "common": root / "common/checkpoint.pt",
                            "condition": name, "manifest": path}
                    parents[name] = train_trial(argparse.Namespace(**fork, output=root / name))
                forks = {}
                for name in ("a", "control"):
                    fork = {**options, "initialize": None, "common": root / name / "checkpoint.pt", "condition": name,
                            "image_manifest": image_path, "steps": 2}
                    forks[name] = train_trial(argparse.Namespace(**fork, output=root / (name + "-image")))
                fork = {**options, "initialize": None, "common": root / "a/checkpoint.pt", "condition": "a",
                        "image_manifest": image_path, "steps": 1}
                train_trial(argparse.Namespace(**fork, output=root / "continued"))
                resumed = {**fork, "common": None, "resume": root / "continued/checkpoint.pt", "steps": 2}
                continued = train_trial(argparse.Namespace(**resumed, output=root / "continued"))
            self.assertEqual(forks["a"]["final_parameters_sha256"], continued["final_parameters_sha256"])
            _, _, whole = load_checkpoint(root / "a-image/checkpoint.pt")
            _, _, split = load_checkpoint(root / "continued/checkpoint.pt")
            self.assertEqual(state_digest(whole["trainer"]), state_digest(split["trainer"]))
            self.assertEqual(state_digest(whole["sources"]), state_digest(split["sources"]))
            self.assertEqual(whole["sources"][LESSON_KEY]["tuition_position"], 4)
            self.assertEqual(len(whole["trainer"]["weights"]), 16)
            self.assertNotIn("text_tuition", whole["trainer"]["weights"])
            traces = [[json.loads(line) for line in (root / (name + "-image/steps.jsonl")).read_text().splitlines()]
                      for name in ("a", "control")]
            for a, control in zip(*traces):
                self.assertEqual(a["background_state_sha256"], control["background_state_sha256"])
                self.assertEqual(a["lesson_inputs"], control["lesson_inputs"])
            for name, result in forks.items():
                self.assertEqual(result["initial_checkpoint_sha256"], parents[name]["checkpoint_sha256"])
                self.assertEqual(result["initial_parameters_sha256"], parents[name]["final_parameters_sha256"])
                self.assertEqual(result["completed_steps"], 2)
                self.assertEqual(result["new_rule_image_training_presentations"], 4)
                self.assertEqual(result["new_rule_image_training_examples"], 4)
                self.assertEqual(result["new_rule_image_evaluation_queries"], 366)
                self.assertEqual(result["parameters"], result["trainable_parameters"])
                self.assertEqual(len(result["source_progress"]), 12)
            from scripts.run_rule_transfer_image_followup import audit_pair
            directories = {name: root / (name + "-image") for name in ("a", "control")}
            audit = audit_pair(root, directories, manifest, file_digest(image_path))
            self.assertTrue(audit["verified"])
            self.assertEqual(audit["matched_shared_prefix_updates"], 2)
            path = directories["control"] / "steps.jsonl"
            corrupted = [json.loads(line) for line in path.read_text().splitlines()]
            corrupted[0]["background_state_sha256"] = "different source sequence"
            path.write_text("\n".join(json.dumps(row) for row in corrupted) + "\n")
            with self.assertRaisesRegex(ValueError, "identical inputs"):
                audit_pair(root, directories, manifest, file_digest(image_path))


class FollowupRunnerTests(unittest.TestCase):
    def test_prepare_command_writes_checked_nested_training_manifests(self):
        from scripts.prepare_rule_transfer_image_followup import main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recipe, _, _, _ = training_fixture(root, twelve=True, per_class=64)
            panel, images, labels = fixture_panel()
            panel["files"] = {"training_images": {"path": "train-images.gz"}}
            (root / "panel.json").write_text(json.dumps(panel))
            (root / "recipe.json").write_text(json.dumps(recipe))
            argv = ["prepare_rule_transfer_image_followup.py", "--panel", str(root / "panel.json"),
                    "--recipe", str(root / "recipe.json"), "--output", str(root / "support"), "--data-root", str(root)]
            with patch("sys.argv", argv), patch("scripts.prepare_rule_transfer_image_followup.load_panel", return_value=(panel, images, labels)):
                main()
            summary = json.loads((root / "support/summary.json").read_text())
            for count in (32, 128, 512):
                path = root / f"support/image-{count:04d}.json"
                manifest = json.loads(path.read_text())
                self.assertEqual(summary[str(count)]["sha256"], file_digest(path))
                self.assertEqual(summary[str(count)]["class_relations_identified_by_transitivity"], 45)
                self.assertEqual(manifest["panel_sha256"], file_digest(root / "panel.json"))
                self.assertEqual(manifest["source_split"], "train")

    def test_extension_requires_remaining_learning_signal_at_the_fixed_boundary(self):
        from scripts.run_rule_transfer_image_followup import extension_reason
        result = {"prerequisites_passed": False, "completed_steps": 1024, "support_accuracy": 1.,
                  "prerequisites": {"new_image_rule": {"accuracy": .7}}}
        previous = {"gates": {"new_image_rule": {"accuracy": .68}}}
        self.assertEqual(extension_reason(result, previous), ["development_gain_at_least_two_points_from_512_to_1024"])
        result["prerequisites"]["new_image_rule"]["accuracy"] = .69
        self.assertEqual(extension_reason(result, previous), [])
        result["support_accuracy"] = .98
        self.assertEqual(extension_reason(result, previous), ["support_accuracy_below_99_percent"])
        result["completed_steps"] = 2048
        self.assertEqual(extension_reason(result, previous), [])
        result.update(completed_steps=1024, prerequisites_passed=True)
        self.assertEqual(extension_reason(result, previous), [])

    def test_holdout_waits_for_all_six_independent_endpoints_and_conditional_extensions(self):
        from scripts.run_rule_transfer_image_followup import main
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            parents, panel, support = root / "parents", root / "panel", root / "support"
            panel.mkdir()
            support.mkdir()
            (panel / "panel.json").write_text("fixture")
            panel_hash = file_digest(panel / "panel.json")
            for name in ("a", "b", "control"):
                (parents / name).mkdir(parents=True)
                (parents / name / "checkpoint.pt").write_text(name)
                (parents / name / "result.json").write_text(json.dumps({"prerequisites_passed": True,
                    "completed_steps": 900, "checkpoint_sha256": file_digest(parents / name / "checkpoint.pt"),
                    "new_rule_image_training_examples": 0, "settings": {"seed": 47}}))
            (root / "audit.json").write_text(json.dumps({"verified": True, "prerequisites_passed": True, "updates_per_condition": 900}))
            lessons = ("digit_names", "old_image_order", "old_text_order", "image_tuition")
            plan = {"panel_sha256": panel_hash, "teacher_budgets": [32, 128, 512],
                    "measurement_updates": [0, 64, 128, 256, 512, 1024], "optimizer": {"learning_rate": 1e-5},
                    "training": {"batches": dict(zip(lessons, [16, 8, 8, 8])), "weights": dict(zip(lessons, [8., 8., 2., 8.]))}}
            (root / "plan.json").write_text(json.dumps(plan))
            for count in (32, 128, 512):
                (support / f"image-{count:04d}.json").write_text(json.dumps({"examples": [None] * count, "panel_sha256": panel_hash}))
            calls = []

            def run(name, arguments):
                args = list(arguments)
                calls.append((name, args))
                value = lambda flag: args[args.index(flag) + 1]
                if name == "train_rule_transfer.py":
                    output = value("--output")
                    output.mkdir(parents=True, exist_ok=True)
                    control = value("--condition") == "control"
                    count = int(value("--image-manifest").stem.split("-")[1])
                    extended = "--resume" in args
                    steps = 2048 if extended else 1024 if control else 64
                    result = {"checkpoint_sha256": output.name, "completed_steps": steps,
                              "prerequisites_passed": not control or extended, "training_seconds": float(steps),
                              "support_accuracy": .98 if count == 32 and control else 1.,
                              "prerequisites": {"new_image_rule": {"accuracy": .7}}}
                    (output / "result.json").write_text(json.dumps(result))
                    (output / "prerequisites").mkdir(exist_ok=True)
                    (output / "prerequisites/step-000512.json").write_text(json.dumps({"gates": {"new_image_rule": {"accuracy": .69}}}))
                elif name == "evaluate_rule_transfer.py":
                    self.assertTrue((root / "output/selection.json").exists())
                    self.assertEqual(len(json.loads((root / "output/selection.json").read_text())["image_endpoints"]), 6)
                    self.assertEqual(value("--split"), "holdout")

            argv = ["run_rule_transfer_image_followup.py", "--parents", str(parents), "--parent-audit", str(root / "audit.json"),
                    "--plan", str(root / "plan.json"), "--panel-directory", str(panel), "--support-directory", str(support),
                    "--work", str(root / "work"), "--output", str(root / "output"), "--archive-prefix", "fixture"]
            with patch("sys.argv", argv), patch("scripts.run_rule_transfer_image_followup.script", side_effect=run), \
                 patch("scripts.run_rule_transfer_image_followup.audit_pair", return_value={"verified": True}):
                main()
            training = [args for name, args in calls if name == "train_rule_transfer.py"]
            self.assertEqual(sum("--common" in args for args in training), 6)
            self.assertEqual(sum("--resume" in args for args in training), 1)
            self.assertEqual(sum(name == "evaluate_rule_transfer.py" for name, _ in calls), 9)
            first_evaluation = next(index for index, (name, _) in enumerate(calls) if name == "evaluate_rule_transfer.py")
            self.assertFalse(any(name == "train_rule_transfer.py" for name, _ in calls[first_evaluation:]))
            self.assertEqual(sum(name == "archive_rule_transfer.py" for name, _ in calls), 6)


if __name__ == "__main__":
    unittest.main()
