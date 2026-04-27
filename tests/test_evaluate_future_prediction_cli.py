import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import evaluate_future_prediction


class EvaluateFuturePredictionCLITest(unittest.TestCase):
    def test_evaluates_typed_event_train_and_eval_paths(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            _write_events(train_path, "train")
            _write_events(eval_path, "eval")

            with redirect_stdout(output):
                evaluate_future_prediction.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--target-role",
                        "consequence",
                        "--condition",
                        "same_modality_negative",
                        "--model-preset",
                        "tiny",
                        "--max-steps",
                        "1",
                        "--context-length",
                        "32",
                        "--batch-size",
                        "2",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        stdout = output.getvalue()
        self.assertIn("intrep future-prediction evaluation", stdout)
        self.assertIn("target_role=consequence", stdout)
        self.assertIn("generalization_eval=true", stdout)
        self.assertEqual(payload["target_role"], "consequence")
        self.assertTrue(payload["generalization_eval"])
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)
        self.assertIn("delta_top1_accuracy", payload)
        self.assertIn("delta_margin", payload)
        self.assertEqual(payload["explicit_negative_rate"], 0.0)
        self.assertEqual(payload["no_negative_case_count"], 0)

    def test_eval_path_absent_reports_train_split_not_generalization(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            metrics_path = root / "metrics.json"
            _write_events(train_path, "train")

            with redirect_stdout(output):
                evaluate_future_prediction.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--target-role",
                        "consequence",
                        "--condition",
                        "same_modality_negative",
                        "--model-preset",
                        "tiny",
                        "--max-steps",
                        "1",
                        "--context-length",
                        "32",
                        "--batch-size",
                        "2",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        stdout = output.getvalue()
        self.assertIn("eval_path=train", stdout)
        self.assertIn("eval_split=train", stdout)
        self.assertIn("generalization_eval=false", stdout)
        self.assertEqual(payload["eval_split"], "train")
        self.assertFalse(payload["generalization_eval"])
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)
        self.assertIn("delta_top1_accuracy", payload)
        self.assertIn("delta_margin", payload)


def _write_events(path: Path, prefix: str) -> None:
    rows = [
        {
            "id": f"{prefix}_ep1_obs",
            "role": "observation",
            "modality": "text",
            "episode_id": f"{prefix}_ep1",
            "time_index": 0,
            "content": "box_a has key ; box_b has coin",
        },
        {
            "id": f"{prefix}_ep1_action",
            "role": "action",
            "modality": "text",
            "episode_id": f"{prefix}_ep1",
            "time_index": 1,
            "content": "open box_a",
        },
        {
            "id": f"{prefix}_ep1_cons",
            "role": "consequence",
            "modality": "text",
            "episode_id": f"{prefix}_ep1",
            "time_index": 2,
            "content": "see key",
        },
        {
            "id": f"{prefix}_ep2_obs",
            "role": "observation",
            "modality": "text",
            "episode_id": f"{prefix}_ep2",
            "time_index": 0,
            "content": "box_a has key ; box_b has coin",
        },
        {
            "id": f"{prefix}_ep2_action",
            "role": "action",
            "modality": "text",
            "episode_id": f"{prefix}_ep2",
            "time_index": 1,
            "content": "open box_b",
        },
        {
            "id": f"{prefix}_ep2_cons",
            "role": "consequence",
            "modality": "text",
            "episode_id": f"{prefix}_ep2",
            "time_index": 2,
            "content": "see coin",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
