import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep import evaluate_image_to_text
from intrep.image_to_text_training import ImageToTextMetrics


class EvaluateImageToTextCLITest(unittest.TestCase):
    def test_builds_image_to_text_config(self) -> None:
        captured_config = None
        captured_train_count = 0

        def fake_train_image_to_text_labels(*, train_examples, eval_examples=None, config):
            nonlocal captured_config, captured_train_count
            captured_config = config
            captured_train_count = len(train_examples)
            return ImageToTextMetrics(
                target="answer_text",
                input_representation="image-patches",
                output_representation="text-tokens",
                train_case_count=1,
                eval_case_count=0,
                train_initial_loss=2.0,
                train_final_loss=1.0,
                eval_initial_loss=None,
                eval_final_loss=None,
                patch_size=config.patch_size,
                max_steps=config.max_steps,
                model_preset=config.model_preset,
            )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            metrics_path = root / "metrics.json"
            _write_image_choice_examples(train_path, root / "images")

            with patch.object(
                evaluate_image_to_text,
                "train_image_to_text_labels",
                fake_train_image_to_text_labels,
            ):
                evaluate_image_to_text.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

        self.assertIsNotNone(captured_config)
        assert captured_config is not None
        self.assertEqual(captured_train_count, 2)
        self.assertEqual(captured_config.patch_size, 4)
        self.assertEqual(captured_config.model_preset, "tiny")

    def test_runs_small_image_to_text_smoke(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            metrics_path = root / "metrics.json"
            _write_image_choice_examples(train_path, root / "images")

            with redirect_stdout(output):
                evaluate_image_to_text.main(
                    [
                        "--train-path",
                        str(train_path),
                        "--max-steps",
                        "1",
                        "--batch-size",
                        "2",
                        "--image-patch-size",
                        "1",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep image to text", output.getvalue())
        self.assertEqual(payload["target"], "answer_text")
        self.assertEqual(payload["output_representation"], "text-tokens")
        self.assertEqual(payload["train_case_count"], 2)


def _write_image_choice_examples(path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / "a.pgm"
    image_b = image_dir / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 0, 0, 0]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 255, 255, 255]))
    rows = [
        {"image_path": str(image_a), "choices": ["dark", "light"], "answer_index": 0},
        {"image_path": str(image_b), "choices": ["dark", "light"], "answer_index": 1},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
