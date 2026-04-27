from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from intrep.experiment_pipeline import run_generated_environment_sweep


class ExperimentPipelineTest(unittest.TestCase):
    def test_generated_environment_sweep_writes_seed_slice_outputs_and_reports(self) -> None:
        calls = []

        def fake_runner(
            documents,
            *,
            eval_documents,
            corpus_label,
            eval_corpus_label,
            training_config,
        ):
            calls.append((training_config.seed, eval_corpus_label, len(documents), len(eval_documents)))
            return {
                "corpus": {"label": corpus_label, "eval_label": eval_corpus_label},
                "training_config": {
                    "max_steps": training_config.max_steps,
                    "seed": training_config.seed,
                },
                "training_loss": {"final_loss": 1.0 + training_config.seed},
                "language_modeling": {
                    "initial_eval_loss": 3.0,
                    "final_eval_loss": 2.0,
                },
                "next_observation": {
                    "before": {"top1_accuracy": 0.25},
                    "after": {"top1_accuracy": 0.5},
                },
                "symbolic_to_natural": {
                    "before": {"top1_accuracy": 0.25},
                    "after": {"top1_accuracy": 0.5},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            result = run_generated_environment_sweep(
                output_dir=directory,
                seeds=[3, 5],
                eval_slices=["generated_seen", "generated_held_out_object"],
                max_steps=4,
                runner=fake_runner,
            )
            root = Path(directory)
            comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(len(calls), 4)
        self.assertEqual(len(result.runs), 4)
        self.assertEqual(result.failures, [])
        self.assertEqual(comparison["run_count"], 4)
        self.assertEqual(failures["failure_count"], 0)
        self.assertEqual(
            comparison["runs"][0]["config"]["corpus"]["train_label"],
            "generated-environment",
        )
        self.assertEqual(
            comparison["runs"][0]["config"]["corpus"]["eval_slice"],
            "generated_seen",
        )
        self.assertEqual(comparison["runs"][0]["config"]["training"]["seed"], 3)

    def test_generated_environment_sweep_records_failures_and_continues(self) -> None:
        def fake_runner(
            documents,
            *,
            eval_documents,
            corpus_label,
            eval_corpus_label,
            training_config,
        ):
            if eval_corpus_label == "generated_held_out_object":
                raise RuntimeError("boom")
            return {
                "training_config": {
                    "max_steps": training_config.max_steps,
                    "seed": training_config.seed,
                },
                "training_loss": {"final_loss": 1.0},
                "language_modeling": {
                    "initial_eval_loss": 3.0,
                    "final_eval_loss": 2.0,
                },
                "next_observation": {
                    "before": {"top1_accuracy": 0.25},
                    "after": {"top1_accuracy": 0.5},
                },
                "symbolic_to_natural": {
                    "before": {"top1_accuracy": 0.25},
                    "after": {"top1_accuracy": 0.5},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            result = run_generated_environment_sweep(
                output_dir=directory,
                seeds=[7],
                eval_slices=["generated_seen", "generated_held_out_object"],
                runner=fake_runner,
            )
            root = Path(directory)
            comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            failures = json.loads((root / "failures.json").read_text(encoding="utf-8"))

        self.assertEqual(len(result.runs), 1)
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(comparison["run_count"], 1)
        self.assertEqual(failures["failure_count"], 1)
        self.assertEqual(failures["failures"][0]["seed"], 7)
        self.assertEqual(failures["failures"][0]["eval_slice"], "generated_held_out_object")
        self.assertEqual(failures["failures"][0]["error_type"], "RuntimeError")

    def test_generated_environment_sweep_reports_metric_regressions(self) -> None:
        def fake_runner(
            documents,
            *,
            eval_documents,
            corpus_label,
            eval_corpus_label,
            training_config,
            distractor_policy,
        ):
            self.assertEqual(distractor_policy, "same_entity")
            return {
                "training_config": {
                    "max_steps": training_config.max_steps,
                    "seed": training_config.seed,
                    "device": training_config.device,
                },
                "training_loss": {"final_loss": 1.0},
                "language_modeling": {
                    "initial_eval_loss": 2.0,
                    "final_eval_loss": 3.0,
                },
                "next_observation": {
                    "before": {"top1_accuracy": 0.75},
                    "after": {"top1_accuracy": 0.25},
                },
                "symbolic_to_natural": {
                    "before": {"top1_accuracy": 0.5},
                    "after": {"top1_accuracy": 0.5},
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            result = run_generated_environment_sweep(
                output_dir=directory,
                seeds=[7],
                eval_slices=["generated_seen"],
                device="auto",
                runner=fake_runner,
            )
            failures = json.loads(
                (Path(directory) / "failures.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(result.runs), 1)
        self.assertEqual(len(result.failures), 2)
        self.assertEqual(failures["failure_count"], 2)
        self.assertEqual(
            {failure["metric"] for failure in failures["failures"]},
            {
                "language_modeling.final_eval_loss",
                "next_observation.after.top1_accuracy",
            },
        )


if __name__ == "__main__":
    unittest.main()
