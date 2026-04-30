import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import evaluate_image_to_text, evaluate_image_to_text_checkpoint


class EvaluateImageToTextCheckpointCLITest(unittest.TestCase):
    def test_evaluates_saved_checkpoint(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            checkpoint_path = root / "model.pt"
            metrics_path = root / "checkpoint-metrics.json"
            _write_image_choice_examples(train_path, root / "images")
            evaluate_image_to_text.main(
                [
                    "--train-path",
                    str(train_path),
                    "--max-steps",
                    "2",
                    "--batch-size",
                    "2",
                    "--image-patch-size",
                    "1",
                    "--checkpoint-path",
                    str(checkpoint_path),
                ]
            )

            with redirect_stdout(output):
                evaluate_image_to_text_checkpoint.main(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--eval-path",
                        str(train_path),
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep image to text checkpoint", output.getvalue())
        self.assertEqual(payload["case_count"], 2)
        self.assertIn("accuracy", payload)


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
