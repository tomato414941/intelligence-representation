import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import evaluate_training_phases


class EvaluateTrainingPhasesCLITest(unittest.TestCase):
    def test_runs_phase_three_smoke(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_examples_path = root / "images.jsonl"
            text_corpus_path = root / "text.txt"
            metrics_path = root / "metrics.json"
            _write_image_choice_examples(image_examples_path, root)
            text_corpus_path.write_text("label: a\nlabel: b\n" * 8, encoding="utf-8")

            with redirect_stdout(output):
                evaluate_training_phases.main(
                    [
                        "--image-examples-path",
                        str(image_examples_path),
                        "--text-corpus-path",
                        str(text_corpus_path),
                        "--prompt",
                        "?",
                        "--image-patch-size",
                        "4",
                        "--context-length",
                        "4",
                        "--batch-size",
                        "2",
                        "--image-max-steps",
                        "1",
                        "--text-max-steps",
                        "2",
                        "--image-text-max-steps",
                        "10",
                        "--learning-rate",
                        "0.05",
                        "--seed",
                        "13",
                        "--device",
                        "cpu",
                        "--metrics-path",
                        str(metrics_path),
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertIn("intrep training phases", output.getvalue())
        self.assertEqual(payload["image_classification"]["train_case_count"], 1)
        self.assertEqual(payload["image_text"]["case_count"], 1)
        self.assertGreater(
            payload["image_text"]["initial_loss"],
            payload["image_text"]["final_loss"],
        )
        self.assertEqual(payload["image_text_choice"]["accuracy"], 1.0)
        self.assertEqual(payload["image_text_choice"]["predicted_indices"], [1])


def _write_image_choice_examples(path: Path, root: Path) -> None:
    image_path = root / "image.pgm"
    image_path.write_text("P2\n4 4\n255\n" + " ".join(["0"] * 16) + "\n", encoding="ascii")
    row = {
        "image_path": str(image_path),
        "choices": ["a", "b"],
        "answer_index": 1,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
