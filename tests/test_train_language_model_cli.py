import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_language_model


class TrainLanguageModelCLITest(unittest.TestCase):
    def test_trains_text_corpus_with_eval_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "checkpoint.pt"
            corpus_path.write_text("first citizen speaks\nsecond citizen replies\n" * 20, encoding="utf-8")

            with redirect_stdout(output):
                train_language_model.main(
                    [
                        "--corpus-path",
                        str(corpus_path),
                        "--metrics-path",
                        str(metrics_path),
                        "--checkpoint-path",
                        str(checkpoint_path),
                        "--context-length",
                        "16",
                        "--batch-size",
                        "2",
                        "--max-steps",
                        "2",
                        "--learning-rate",
                        "0.01",
                        "--device",
                        "cpu",
                    ]
                )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            checkpoint_exists = checkpoint_path.exists()

        self.assertIn("intrep train language model", output.getvalue())
        self.assertTrue(checkpoint_exists)
        self.assertEqual(payload["schema_version"], "intrep.language_model_run.v1")
        self.assertEqual(payload["metrics"]["eval_split"], "held_out")
        self.assertTrue(payload["metrics"]["generalization_eval"])
        self.assertGreater(payload["train_char_count"], payload["eval_char_count"])
        self.assertGreater(payload["result"]["token_count"], 0)

    def test_split_text_corpus_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "corpus must not be empty"):
            train_language_model.split_text_corpus("", eval_ratio=0.1)


if __name__ == "__main__":
    unittest.main()
