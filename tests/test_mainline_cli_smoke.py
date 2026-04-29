import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import evaluate_future_prediction, train_signal_text
from intrep.generated_environment_signal_corpus import write_generated_environment_signal_jsonl


class MainlineCLISmokeTest(unittest.TestCase):
    def test_signal_text_training_and_future_prediction_cli_smoke(self) -> None:
        train_output = io.StringIO()
        eval_output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.signals.jsonl"
            eval_path = root / "eval.signals.jsonl"
            train_summary_path = root / "train-summary.json"
            eval_metrics_path = root / "future-metrics.json"
            write_generated_environment_signal_jsonl(
                train_path,
                eval_path,
                eval_slice="same_history_different_action",
                train_size=2,
                eval_size=2,
                seed=3,
            )

            with redirect_stdout(train_output):
                train_signal_text.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--max-steps",
                        "1",
                        "--context-length",
                        "16",
                        "--batch-size",
                        "1",
                        "--model-preset",
                        "tiny",
                        "--run-summary-path",
                        str(train_summary_path),
                    ]
                )
            with redirect_stdout(eval_output):
                evaluate_future_prediction.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--target-channel",
                        "consequence",
                        "--condition",
                        "same_modality_negative",
                        "--max-steps",
                        "1",
                        "--context-length",
                        "16",
                        "--batch-size",
                        "1",
                        "--model-preset",
                        "tiny",
                        "--rendering",
                        "payload",
                        "--metrics-path",
                        str(eval_metrics_path),
                    ]
                )

            train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
            eval_metrics = json.loads(eval_metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep signal-text training", train_output.getvalue())
        self.assertEqual(train_summary["kind"], "train_signal_text")
        self.assertEqual(train_summary["metrics"]["training_loss"]["steps"], 1)
        self.assertTrue(train_summary["metrics"]["training_loss"]["generalization_eval"])
        self.assertIn("intrep future-prediction evaluation", eval_output.getvalue())
        self.assertEqual(eval_metrics["target_channel"], "consequence")
        self.assertEqual(eval_metrics["condition"], "same_modality_negative")
        self.assertEqual(eval_metrics["train_case_count"], 2)
        self.assertEqual(eval_metrics["eval_case_count"], 2)
        self.assertTrue(eval_metrics["generalization_eval"])


if __name__ == "__main__":
    unittest.main()
