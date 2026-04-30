import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_text_image_shared_core
from intrep.image_classification import FASHION_MNIST_LABELS


class TrainTextImageSharedCoreCLITest(unittest.TestCase):
    def test_trains_shared_core_and_writes_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            image_path = root / "images.jsonl"
            metrics_path = root / "metrics.json"
            corpus_path.write_text("alpha beta gamma alpha beta gamma\n" * 30, encoding="utf-8")
            _write_image_examples(image_path, root / "images")

            with redirect_stdout(output):
                train_text_image_shared_core.main(
                    [
                        "--corpus-path",
                        str(corpus_path),
                        "--image-train-path",
                        str(image_path),
                        "--image-eval-path",
                        str(image_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--text-context-length",
                        "8",
                        "--image-patch-size",
                        "1",
                        "--max-steps",
                        "2",
                        "--batch-size",
                        "2",
                        "--device",
                        "cpu",
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep text image shared core", output.getvalue())
        self.assertEqual(payload["schema_version"], "intrep.text_image_shared_core_run.v1")
        self.assertTrue(payload["metrics"]["shared_core"])
        self.assertEqual(payload["metrics"]["image_train_case_count"], 2)
        self.assertEqual(payload["metrics"]["image_eval_case_count"], 2)
        self.assertGreater(payload["metrics"]["text_token_count"], 8)


def _write_image_examples(path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    image_a = image_dir / "a.pgm"
    image_b = image_dir / "b.pgm"
    image_a.write_bytes(b"P5\n2 2\n255\n" + bytes([0, 255, 0, 255]))
    image_b.write_bytes(b"P5\n2 2\n255\n" + bytes([255, 0, 255, 0]))
    rows = [
        {
            "image_path": str(image_a),
            "choices": list(FASHION_MNIST_LABELS),
            "answer_index": 9,
        },
        {
            "image_path": str(image_b),
            "choices": list(FASHION_MNIST_LABELS),
            "answer_index": 0,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
