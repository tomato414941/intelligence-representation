import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import evaluate_image_to_text, predict_image_text_choice


class PredictImageTextChoiceCLITest(unittest.TestCase):
    def test_predicts_from_saved_checkpoint(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            image_dir = root / "images"
            checkpoint_path = root / "model.pt"
            output_path = root / "prediction.json"
            image_path = image_dir / "a.pgm"
            _write_image_choice_examples(train_path, image_dir)
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
                predict_image_text_choice.main(
                    [
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--image-path",
                        str(image_path),
                        "--choice",
                        "dark",
                        "--choice",
                        "light",
                        "--output-path",
                        str(output_path),
                    ]
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("intrep image text choice prediction", output.getvalue())
        self.assertIn(payload["predicted_index"], (0, 1))
        self.assertIn(payload["predicted_choice"], ("dark", "light"))
        self.assertEqual(len(payload["losses"]), 2)


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
