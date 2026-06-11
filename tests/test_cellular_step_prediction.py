import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from intrep import train_cellular_step_prediction
from intrep.problems.cellular_step_prediction import evaluate
from intrep.problems.cellular_step_prediction.checkpoint import load_cellular_step_checkpoint
from intrep.problems.cellular_step_prediction.dataset import CellularStepPredictionDataset
from intrep.problems.cellular_step_prediction.metrics import cellular_step_prediction_scores
from intrep.problems.cellular_step_prediction.training import (
    CellularStepPredictionConfig,
    train_cellular_step_predictor,
)
from intrep.representation.assemblies.cellular_step_prediction import CellularStepPredictionModel
from intrep.worlds.cellular.world import LIFE_RULE, generate_cellular_transitions


def _tiny_config(max_steps: int = 50) -> CellularStepPredictionConfig:
    return CellularStepPredictionConfig(
        max_steps=max_steps,
        batch_size=8,
        embedding_dim=32,
        num_heads=2,
        hidden_dim=64,
        num_layers=1,
        device="cpu",
    )


class CellularStepPredictionModelTest(unittest.TestCase):
    def test_returns_per_cell_logits(self) -> None:
        model = CellularStepPredictionModel(
            height=4, width=5, embedding_dim=8, num_heads=2, hidden_dim=16, num_layers=1
        )

        logits = model(torch.zeros((3, 1, 4, 5)))

        self.assertEqual(tuple(logits.shape), (3, 20, 2))


class CellularStepPredictionScoresTest(unittest.TestCase):
    def test_copy_strategy_scores_zero_on_changed_cells(self) -> None:
        transitions = generate_cellular_transitions(LIFE_RULE, width=6, height=4, count=10, seed=5)
        dataset = CellularStepPredictionDataset(transitions)
        copy_predictions = torch.stack([dataset[index][0].reshape(-1).long() for index in range(len(dataset))])

        scores = cellular_step_prediction_scores(copy_predictions, transitions)

        self.assertEqual(scores.changed_cell_accuracy, 0.0)
        self.assertEqual(scores.unchanged_cell_accuracy, 1.0)


class CellularStepPredictionTrainingTest(unittest.TestCase):
    def test_training_reduces_loss(self) -> None:
        transitions = generate_cellular_transitions(LIFE_RULE, width=5, height=4, count=32, seed=9)

        artifacts = train_cellular_step_predictor(transitions, config=_tiny_config(max_steps=200))

        self.assertEqual(artifacts.result.train_case_count, 32)
        self.assertLess(artifacts.result.final_loss, artifacts.result.initial_loss)
        self.assertIsNotNone(artifacts.result.train_scores.changed_cell_accuracy)
        self.assertIsNotNone(artifacts.result.train_scores.unchanged_cell_accuracy)


_TINY_CLI_ARGS = [
    "--grid-width",
    "5",
    "--grid-height",
    "4",
    "--train-count",
    "16",
    "--max-steps",
    "1",
    "--warmup-steps",
    "0",
    "--embedding-dim",
    "16",
    "--num-heads",
    "2",
    "--hidden-dim",
    "32",
    "--num-layers",
    "1",
    "--device",
    "cpu",
]


class TrainCellularStepPredictionCLITest(unittest.TestCase):
    def test_writes_checkpoint_and_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "cellular.pt"
            metrics_path = root / "cellular.json"

            with redirect_stdout(output):
                train_cellular_step_prediction.main(
                    ["--checkpoint-path", str(checkpoint_path), "--metrics-path", str(metrics_path)]
                    + _TINY_CLI_ARGS
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_cellular_step_checkpoint(checkpoint_path, device="cpu")

        self.assertIn("intrep train cellular step prediction", output.getvalue())
        self.assertEqual(payload["schema_version"], "intrep.cellular_step_prediction_run.v1")
        self.assertEqual(payload["world"]["rule"], {"birth": [3], "survival": [2, 3]})
        self.assertEqual(payload["result"]["train_case_count"], 16)
        self.assertIn("changed_cell_accuracy", payload["result"]["train_scores"])
        self.assertEqual(checkpoint.rule, LIFE_RULE)
        self.assertEqual(checkpoint.grid_size, (4, 5))
        self.assertEqual(checkpoint.train_data["count"], 16)
        logits = checkpoint.model(torch.zeros((1, 1, 4, 5)))
        self.assertEqual(tuple(logits.shape), (1, 20, 2))


class EvaluateCellularStepPredictionCLITest(unittest.TestCase):
    def test_evaluates_on_unseen_states_with_copy_baseline(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "cellular.pt"
            train_metrics_path = root / "train.json"
            eval_metrics_path = root / "eval.json"

            with redirect_stdout(output):
                train_cellular_step_prediction.main(
                    ["--checkpoint-path", str(checkpoint_path), "--metrics-path", str(train_metrics_path)]
                    + _TINY_CLI_ARGS
                )
                evaluate.main(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--metrics-path",
                        str(eval_metrics_path),
                        "--eval-count",
                        "8",
                        "--eval-seed",
                        "1000",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep evaluate cellular step prediction", output.getvalue())
        self.assertEqual(payload["schema_version"], "intrep.cellular_step_prediction_eval.v1")
        self.assertEqual(payload["eval_data"], {"count": 8, "seed": 1000, "alive_probability": 0.5})
        self.assertIn("changed_cell_accuracy", payload["model_scores"])
        self.assertEqual(payload["copy_baseline_scores"]["changed_cell_accuracy"], 0.0)
        self.assertEqual(payload["copy_baseline_scores"]["unchanged_cell_accuracy"], 1.0)

    def test_rejects_eval_seeds_overlapping_training_data(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "cellular.pt"

            with redirect_stdout(io.StringIO()):
                train_cellular_step_prediction.main(
                    ["--checkpoint-path", str(checkpoint_path), "--metrics-path", str(root / "t.json")]
                    + _TINY_CLI_ARGS
                )

            with self.assertRaises(ValueError):
                evaluate.main(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--metrics-path",
                        str(root / "e.json"),
                        "--eval-seed",
                        "10",
                        "--device",
                        "cpu",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
