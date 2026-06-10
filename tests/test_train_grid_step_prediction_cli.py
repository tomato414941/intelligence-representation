import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_grid_step_prediction
from intrep.problems.grid_step_prediction.checkpoint import load_grid_core_checkpoint


class TrainGridStepPredictionCLITest(unittest.TestCase):
    def test_trains_grid_step_predictor_with_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "grid-step.json"

            with redirect_stdout(output):
                train_grid_step_prediction.main(
                    [
                        "--metrics-path",
                        str(metrics_path),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "5",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep train grid step prediction", output.getvalue())
        self.assertEqual(payload["schema_version"], "intrep.grid_step_prediction_run.v2")
        self.assertEqual(payload["world"]["kind"], "grid_world")
        self.assertEqual(payload["train_case_count"], 25)
        self.assertEqual(payload["eval_case_count"], 0)
        self.assertEqual(payload["training_config"]["max_steps"], 1)
        self.assertEqual(payload["result"]["train_case_count"], 25)
        self.assertEqual(payload["result"]["eval_case_count"], 0)
        self.assertIn("final_next_observation_loss", payload["result"])
        self.assertIn("final_reward_loss", payload["result"])
        self.assertIn("final_terminated_loss", payload["result"])
        self.assertIn("copy", payload["baselines"]["train"])

    def test_can_hold_out_agent_cell_for_eval(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "grid-step-held-out.json"

            with redirect_stdout(output):
                train_grid_step_prediction.main(
                    [
                        "--metrics-path",
                        str(metrics_path),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "5",
                        "--eval-agent-cell",
                        "0",
                        "2",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("eval_cases=5", output.getvalue())
        self.assertEqual(payload["held_out_agent_cells"], [{"row": 0, "col": 2}])
        self.assertEqual(payload["train_case_count"], 20)
        self.assertEqual(payload["eval_case_count"], 5)
        self.assertEqual(payload["result"]["train_case_count"], 20)
        self.assertEqual(payload["result"]["eval_case_count"], 5)
        self.assertIsNotNone(payload["result"]["eval_next_agent_cell_accuracy"])
        self.assertIn("next_agent_cell_accuracy", payload["baselines"]["eval"]["copy"])

    def test_can_save_grid_core_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "grid-step.json"
            checkpoint_path = root / "grid-core.pt"

            with redirect_stdout(io.StringIO()):
                train_grid_step_prediction.main(
                    [
                        "--metrics-path",
                        str(metrics_path),
                        "--core-checkpoint-path",
                        str(checkpoint_path),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "5",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint = load_grid_core_checkpoint(checkpoint_path, device="cpu")

        self.assertEqual(payload["core_checkpoint_path"], str(checkpoint_path))
        self.assertEqual(checkpoint.config.max_steps, 1)
        self.assertEqual(checkpoint.grid_size, (2, 3))
        self.assertEqual(checkpoint.metrics["train_case_count"], 25)


class TrainGridStepPredictionDefaultFitTest(unittest.TestCase):
    def test_default_config_fits_transition_table(self) -> None:
        """Anchor for recorded claims: the CLI defaults must fit training.

        Guards against silent behavioral drift in defaults or shared training
        code invalidating recorded results (as happened on 2026-05-05).
        """
        with TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "grid-default-fit.json"

            with redirect_stdout(io.StringIO()):
                train_grid_step_prediction.main(
                    ["--metrics-path", str(metrics_path), "--device", "cpu"]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertGreaterEqual(payload["result"]["next_agent_cell_accuracy"], 0.9)
        self.assertGreaterEqual(payload["result"]["per_cell_accuracy"], 0.95)


if __name__ == "__main__":
    unittest.main()
