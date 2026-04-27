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
