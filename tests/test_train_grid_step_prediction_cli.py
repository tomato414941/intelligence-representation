import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_grid_step_prediction


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
        self.assertEqual(payload["schema_version"], "intrep.grid_step_prediction_run.v1")
        self.assertEqual(payload["world"]["kind"], "grid_world")
        self.assertEqual(payload["train_case_count"], 25)
        self.assertEqual(payload["training_config"]["max_steps"], 1)
        self.assertEqual(payload["result"]["train_case_count"], 25)
        self.assertIn("final_next_cell_loss", payload["result"])
        self.assertIn("final_reward_loss", payload["result"])
        self.assertIn("final_terminated_loss", payload["result"])


if __name__ == "__main__":
    unittest.main()
