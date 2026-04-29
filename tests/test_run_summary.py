from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from intrep.causal_text_model import build_causal_text_config
from intrep.language_modeling_training import LanguageModelingTrainingConfig
from intrep.run_summary import (
    RUN_COMPARISON_SCHEMA,
    aggregate_json_outputs,
    build_run_summary,
    compare_json_outputs,
    main,
    normalize_existing_json,
)


class RunSummaryTest(unittest.TestCase):
    def test_build_run_summary_keeps_configs_and_metrics(self) -> None:
        payload = build_run_summary(
            kind="current_experiment",
            run_id="run-1",
            corpus={"train": {"label": "train.jsonl"}},
            training_config=LanguageModelingTrainingConfig(max_steps=3),
            model_config=build_causal_text_config(
                preset="tiny",
                vocab_size=256,
                context_length=64,
            ),
            training_loss={"final_loss": 2.0, "best_loss": 1.75},
            language_modeling={"final_eval_perplexity": 12.0},
            elapsed_seconds=1.25,
        )

        self.assertEqual(payload["schema_version"], "intrep.run_summary.v1")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["elapsed_seconds"], 1.25)
        self.assertEqual(payload["config"]["training"]["max_steps"], 3)
        self.assertEqual(payload["config"]["training"]["device"], "cpu")
        self.assertEqual(payload["config"]["model"]["embedding_dim"], 8)
        self.assertEqual(payload["metrics"]["language_modeling"]["final_eval_perplexity"], 12.0)
        self.assertEqual(payload["metrics"]["training_loss"]["final_loss"], 2.0)
        self.assertEqual(payload["metrics"]["training_loss"]["best_loss"], 1.75)
        self.assertNotIn("symbolic_to_natural", payload["metrics"])

    def test_normalizes_current_run_summary_json(self) -> None:
        original = build_run_summary(
            kind="current_experiment",
            run_id="run-1",
            training_loss={"final_loss": 2.0},
        )

        payload = normalize_existing_json(original, source_path="summary.json")

        self.assertEqual(payload, original)

    def test_rejects_legacy_summary_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected intrep.run_summary.v1"):
            normalize_existing_json(
                {
                    "training_loss": {"final_loss": 2.0},
                    "language_modeling": {"final_eval_perplexity": 10.0},
                }
            )

    def test_aggregate_json_outputs_wraps_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        corpus={"label": "train", "eval_label": "eval"},
                        training_loss={"final_loss": 2.0},
                        language_modeling={"final_eval_perplexity": 10.0},
                    )
                ),
                encoding="utf-8",
            )

            payload = aggregate_json_outputs([summary_path])

        self.assertEqual(payload["schema_version"], "intrep.run_collection.v1")
        self.assertEqual(payload["run_count"], 1)
        self.assertEqual(payload["runs"][0]["kind"], "current_experiment")

    def test_cli_aggregate_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            output_path = root / "runs.json"
            summary_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        corpus={"label": "train", "eval_label": "eval"},
                        training_loss={"final_loss": 2.0},
                        language_modeling={"final_eval_perplexity": 10.0},
                    )
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(["aggregate", "--input", str(summary_path), "--output", str(output_path)])
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_count"], 1)

    def test_compare_json_outputs_extracts_default_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.json"
            second_path = root / "second.json"
            first_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        run_id="first-run",
                        training_loss={"final_loss": 4.0, "loss_reduction_ratio": 0.2},
                        language_modeling={"final_eval_loss": 3.0, "final_eval_perplexity": 10.0},
                    )
                ),
                encoding="utf-8",
            )
            second_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        run_id="second-run",
                        training_loss={"final_loss": 3.0, "loss_reduction_ratio": 0.3},
                        language_modeling={"final_eval_loss": 2.5, "final_eval_perplexity": 8.0},
                    )
                ),
                encoding="utf-8",
            )

            payload = compare_json_outputs([first_path, second_path])

        self.assertEqual(payload["schema_version"], RUN_COMPARISON_SCHEMA)
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["runs"][0]["run_id"], "first-run")
        self.assertEqual(
            payload["runs"][0]["values"]["metrics.language_modeling.final_eval_loss"],
            3.0,
        )
        self.assertEqual(
            payload["metrics"],
            [
                "metrics.language_modeling.final_eval_loss",
                "metrics.language_modeling.final_eval_perplexity",
                "metrics.language_modeling.final_train_loss",
                "metrics.training_loss.final_loss",
                "metrics.training_loss.best_loss",
                "metrics.training_loss.loss_reduction",
                "metrics.training_loss.loss_reduction_ratio",
                "elapsed_seconds",
            ],
        )

    def test_compare_json_outputs_includes_corpus_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        corpus={
                            "train": {"label": "generated-environment"},
                            "eval": {"label": "generated_held_out_container"},
                            "eval_slice": "generated_held_out_container",
                        },
                        training_config=LanguageModelingTrainingConfig(max_steps=3, seed=11),
                    )
                ),
                encoding="utf-8",
            )

            payload = compare_json_outputs([summary_path])

        config = payload["runs"][0]["config"]
        self.assertEqual(config["corpus"]["train_label"], "generated-environment")
        self.assertEqual(config["corpus"]["eval_label"], "generated_held_out_container")
        self.assertEqual(config["corpus"]["eval_slice"], "generated_held_out_container")
        self.assertEqual(config["training"]["seed"], 11)

    def test_compare_json_outputs_sorts_missing_values_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_path = root / "missing.json"
            scored_path = root / "scored.json"
            missing_path.write_text(json.dumps(build_run_summary(kind="current_experiment", run_id="missing")), encoding="utf-8")
            scored_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        run_id="scored",
                        language_modeling={"final_eval_loss": 1.5},
                    )
                ),
                encoding="utf-8",
            )

            payload = compare_json_outputs(
                [missing_path, scored_path],
                metrics=["metrics.language_modeling.final_eval_loss"],
                sort_by="metrics.language_modeling.final_eval_loss",
            )

        self.assertEqual([run["run_id"] for run in payload["runs"]], ["scored", "missing"])

    def test_cli_compare_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "comparison.json"
            baseline_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        training_loss={"final_loss": 2.0},
                        language_modeling={"final_eval_perplexity": 6.0},
                    )
                ),
                encoding="utf-8",
            )
            candidate_path.write_text(
                json.dumps(
                    build_run_summary(
                        kind="current_experiment",
                        training_loss={"final_loss": 1.5},
                        language_modeling={"final_eval_perplexity": 4.0},
                    )
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(
                    [
                        "compare",
                        "--input",
                        str(baseline_path),
                        "--input",
                        str(candidate_path),
                        "--metric",
                        "metrics.training_loss.final_loss",
                        "--output",
                        str(output_path),
                    ]
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], RUN_COMPARISON_SCHEMA)
        self.assertEqual(payload["run_count"], 2)
        self.assertEqual(payload["runs"][0]["values"]["metrics.training_loss.final_loss"], 2.0)


if __name__ == "__main__":
    unittest.main()
