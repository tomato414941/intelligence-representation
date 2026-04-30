import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_language_model
from intrep.text_tokenizer import save_text_tokenizer, train_byte_pair_tokenizer


class TrainLanguageModelCLITest(unittest.TestCase):
    def test_trains_text_corpus_with_eval_metrics(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            metrics_path = root / "metrics.json"
            checkpoint_path = root / "checkpoint.pt"
            corpus_path.write_text("first citizen speaks\nsecond citizen replies\n" * 80, encoding="utf-8")

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
        self.assertEqual(payload["corpus_paths"], [str(corpus_path)])
        self.assertEqual(payload["metrics"]["eval_split"], "held_out")
        self.assertTrue(payload["metrics"]["generalization_eval"])
        self.assertGreater(payload["train_char_count"], payload["eval_char_count"])
        self.assertGreater(payload["result"]["token_count"], 0)

    def test_reads_multiple_text_corpora_with_separator(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            alpha_path = root / "alpha.txt"
            beta_path = root / "beta.txt"
            alpha_path.write_text("alpha alpha", encoding="utf-8")
            beta_path.write_text("beta beta", encoding="utf-8")

            corpus = train_language_model.read_text_corpora(
                [alpha_path, beta_path],
                seed=1,
            )

        self.assertIn("alpha alpha", corpus)
        self.assertIn("beta beta", corpus)
        self.assertIn("<|endoftext|>", corpus)

    def test_reads_multiple_text_corpora_with_per_corpus_eval_split(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            alpha_path = root / "alpha.txt"
            beta_path = root / "beta.txt"
            alpha_path.write_text("a" * 10, encoding="utf-8")
            beta_path.write_text("b" * 10, encoding="utf-8")

            train_text, eval_text = train_language_model.read_split_text_corpora(
                [alpha_path, beta_path],
                eval_ratio=0.2,
                seed=1,
            )

        self.assertEqual(train_text.count("a"), 8)
        self.assertEqual(train_text.count("b"), 8)
        self.assertEqual(eval_text.count("a"), 2)
        self.assertEqual(eval_text.count("b"), 2)

    def test_trains_with_saved_tokenizer(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            tokenizer_path = root / "tokenizer.json"
            metrics_path = root / "metrics.json"
            corpus_path.write_text("red green blue red green blue\n" * 20, encoding="utf-8")
            tokenizer = train_byte_pair_tokenizer(corpus_path.read_text(encoding="utf-8"), vocab_size=270)
            save_text_tokenizer(tokenizer_path, tokenizer)

            train_language_model.main(
                [
                    "--corpus-path",
                    str(corpus_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--tokenizer-path",
                    str(tokenizer_path),
                    "--context-length",
                    "8",
                    "--batch-size",
                    "2",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["training_config"]["tokenizer"], "byte-pair")
        self.assertEqual(payload["tokenizer"]["source"], str(tokenizer_path))
        self.assertEqual(payload["tokenizer"]["payload"]["kind"], "byte-pair")
        self.assertGreaterEqual(payload["tokenizer"]["payload"]["vocab_size"], 267)
        self.assertLessEqual(payload["tokenizer"]["payload"]["vocab_size"], 270)
        self.assertGreater(payload["result"]["token_count"], 0)

    def test_trains_with_byte_pair_tokenizer(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            metrics_path = root / "metrics.json"
            corpus_path.write_text("red green blue red green blue\n" * 20, encoding="utf-8")

            train_language_model.main(
                [
                    "--corpus-path",
                    str(corpus_path),
                    "--metrics-path",
                    str(metrics_path),
                    "--tokenizer",
                    "byte-pair",
                    "--tokenizer-vocab-size",
                    "270",
                    "--context-length",
                    "8",
                    "--batch-size",
                    "2",
                    "--max-steps",
                    "1",
                    "--device",
                    "cpu",
                ]
            )

            payload = json.loads(metrics_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["tokenizer"]["source"], "trained")
        self.assertEqual(payload["tokenizer"]["payload"]["kind"], "byte-pair")
        self.assertGreater(payload["result"]["token_count"], 0)

    def test_split_text_corpus_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "corpus must not be empty"):
            train_language_model.split_text_corpus("", eval_ratio=0.1)


if __name__ == "__main__":
    unittest.main()
