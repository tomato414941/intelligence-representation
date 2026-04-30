import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep import train_text_tokenizer
from intrep.text_tokenizer import BytePairTokenizer, SimpleBytePairTokenizer, load_text_tokenizer


class TrainTextTokenizerCLITest(unittest.TestCase):
    def test_trains_and_saves_simple_byte_pair_tokenizer(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            tokenizer_path = root / "tokenizer.json"
            corpus_path.write_text("hello hello hello world\n" * 8, encoding="utf-8")

            with redirect_stdout(output):
                train_text_tokenizer.main(
                    [
                        "--corpus-path",
                        str(corpus_path),
                        "--tokenizer-path",
                        str(tokenizer_path),
                        "--tokenizer",
                        "simple-byte-pair",
                        "--tokenizer-vocab-size",
                        "260",
                    ]
                )

            tokenizer = load_text_tokenizer(tokenizer_path)

        self.assertIn("intrep train text tokenizer", output.getvalue())
        self.assertIsInstance(tokenizer, SimpleBytePairTokenizer)
        self.assertEqual(tokenizer.vocab_size, 260)
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello world")), "hello world")

    def test_trains_and_saves_byte_pair_tokenizer_by_default(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.txt"
            tokenizer_path = root / "tokenizer.json"
            corpus_path.write_text("hello hello hello world\n" * 8, encoding="utf-8")

            with redirect_stdout(output):
                train_text_tokenizer.main(
                    [
                        "--corpus-path",
                        str(corpus_path),
                        "--tokenizer-path",
                        str(tokenizer_path),
                        "--tokenizer-vocab-size",
                        "270",
                    ]
                )

            tokenizer = load_text_tokenizer(tokenizer_path)

        self.assertIn("tokenizer=byte-pair", output.getvalue())
        self.assertIsInstance(tokenizer, BytePairTokenizer)
        self.assertGreaterEqual(tokenizer.vocab_size, 267)
        self.assertLessEqual(tokenizer.vocab_size, 270)
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello unseen")), "hello unseen")


if __name__ == "__main__":
    unittest.main()
