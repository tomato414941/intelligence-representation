from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from intrep.gpt_model import build_gpt_config
from intrep.gpt_training import GPTTrainingConfig
from intrep.run_summary import (
    aggregate_json_outputs,
    build_run_summary,
    main,
    normalize_existing_json,
)


class RunSummaryTest(unittest.TestCase):
    def test_build_run_summary_keeps_configs_and_metrics(self) -> None:
        payload = build_run_summary(
            kind="current_experiment",
            run_id="run-1",
            corpus={"train": {"label": "train.jsonl"}},
            training_config=GPTTrainingConfig(max_steps=3),
            model_config=build_gpt_config(
                preset="tiny",
                vocab_size=256,
                context_length=64,
            ),
            training_loss={"final_loss": 2.0},
            language_modeling={"final_eval_perplexity": 12.0},
            elapsed_seconds=1.25,
        )

        self.assertEqual(payload["schema_version"], "intrep.run_summary.v1")
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["elapsed_seconds"], 1.25)
        self.assertEqual(payload["config"]["training"]["max_steps"], 3)
        self.assertEqual(payload["config"]["model"]["embedding_dim"], 8)
        self.assertEqual(payload["metrics"]["language_modeling"]["final_eval_perplexity"], 12.0)

    def test_normalizes_current_experiment_json(self) -> None:
        payload = normalize_existing_json(
            {
                "corpus": {"label": "train", "eval_label": "eval"},
                "training_config": {"max_steps": 2},
                "model_config": {"embedding_dim": 8},
                "training_loss": {"final_loss": 2.0},
                "language_modeling": {"final_eval_perplexity": 10.0},
                "next_observation": {"status": "skipped"},
            },
            source_path="summary.json",
        )

        self.assertEqual(payload["kind"], "current_experiment")
        self.assertEqual(payload["source"]["path"], "summary.json")
        self.assertEqual(payload["metrics"]["training_loss"]["final_loss"], 2.0)

    def test_normalizes_loss_history_json_with_language_metrics(self) -> None:
        payload = normalize_existing_json(
            {
                "steps": 3,
                "batch_stride": None,
                "loss_history": [4.0, 3.0, 2.0],
                "initial_loss": 4.0,
                "final_loss": 2.0,
                "best_loss": 2.0,
                "initial_train_loss": 4.0,
                "final_train_loss": 2.0,
                "initial_eval_loss": None,
                "final_eval_loss": None,
            }
        )

        self.assertEqual(payload["kind"], "train_gpt_loss_history")
        self.assertAlmostEqual(
            payload["metrics"]["language_modeling"]["final_train_perplexity"],
            7.38905609893065,
        )

    def test_aggregate_json_outputs_wraps_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "corpus": {"label": "train", "eval_label": "eval"},
                        "training_config": {"max_steps": 2},
                        "training_loss": {"final_loss": 2.0},
                        "language_modeling": {"final_eval_perplexity": 10.0},
                        "next_observation": {"status": "skipped"},
                    }
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
                    {
                        "corpus": {"label": "train", "eval_label": "eval"},
                        "training_config": {"max_steps": 2},
                        "training_loss": {"final_loss": 2.0},
                        "language_modeling": {"final_eval_perplexity": 10.0},
                        "next_observation": {"status": "skipped"},
                    }
                ),
                encoding="utf-8",
            )

            with redirect_stdout(io.StringIO()):
                main(["aggregate", "--input", str(summary_path), "--output", str(output_path)])
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["run_count"], 1)


if __name__ == "__main__":
    unittest.main()
