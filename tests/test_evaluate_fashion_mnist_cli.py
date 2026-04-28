import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep import evaluate_fashion_mnist


class EvaluateFashionMNISTCLITest(unittest.TestCase):
    def test_builds_fashion_mnist_future_prediction_config(self) -> None:
        captured_config = None

        def fake_run_future_prediction_evaluation(config) -> None:
            nonlocal captured_config
            captured_config = config

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            train_path.write_text('{"channel":"label","payload":"0:T-shirt/top"}\n', encoding="utf-8")
            eval_path.write_text('{"channel":"label","payload":"1:Trouser"}\n', encoding="utf-8")

            with patch.object(
                evaluate_fashion_mnist,
                "run_future_prediction_evaluation",
                fake_run_future_prediction_evaluation,
            ):
                evaluate_fashion_mnist.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

        self.assertIsNotNone(captured_config)
        assert captured_config is not None
        self.assertEqual(captured_config.target_channel, "label")
        self.assertEqual(captured_config.rendering, "image-tokens")
        self.assertEqual(captured_config.image_patch_size, 4)
        self.assertEqual(captured_config.image_channel_bins, 4)
        self.assertEqual(captured_config.image_token_format, "flat")
        self.assertEqual(captured_config.max_negatives, 3)
        self.assertEqual(captured_config.context_length, 96)
        self.assertEqual(captured_config.model_preset, "tiny")

    def test_runs_small_image_label_smoke(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            eval_path = root / "eval.jsonl"
            metrics_path = root / "metrics.json"
            _write_image_label_events(train_path, root / "train-images", "train")
            _write_image_label_events(eval_path, root / "eval-images", "eval")

            with redirect_stdout(output):
                evaluate_fashion_mnist.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--eval-path",
                        str(eval_path),
                        "--max-steps",
                        "1",
                        "--context-length",
                        "32",
                        "--batch-size",
                        "2",
                        "--image-patch-size",
                        "1",
                        "--image-token-format",
                        "grid",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("target_channel=label", output.getvalue())
        self.assertIn("rendering=image-tokens", output.getvalue())
        self.assertEqual(payload["target_channel"], "label")
        self.assertEqual(payload["rendering"], "image-tokens")
        self.assertEqual(payload["train_case_count"], 2)
        self.assertEqual(payload["eval_case_count"], 2)


def _write_image_label_events(path: Path, image_dir: Path, prefix: str) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / f"{prefix}_a.pgm"
    image_b = image_dir / f"{prefix}_b.pgm"
    image_a.write_bytes(b"P5\n2 1\n255\n" + bytes([0, 255]))
    image_b.write_bytes(b"P5\n2 1\n255\n" + bytes([255, 0]))
    rows = [
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_a.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "9:Ankle boot"},
        {
            "channel": "image",
            "payload_ref": {
                "uri": image_b.as_uri(),
                "media_type": "image/x-portable-graymap",
            },
        },
        {"channel": "label", "payload": "0:T-shirt/top"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
