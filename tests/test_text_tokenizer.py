import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.byte_tokenizer import ByteTokenizer
from intrep.text_tokenizer import (
    BytePairTokenizer,
    build_text_tokenizer,
    load_text_tokenizer,
    save_text_tokenizer,
    text_tokenizer_from_payload,
    text_tokenizer_to_payload,
    train_byte_pair_tokenizer,
)


class TextTokenizerTest(unittest.TestCase):
    def test_build_text_tokenizer_uses_byte_pair_default(self) -> None:
        tokenizer = build_text_tokenizer("hello hello", vocab_size=270)

        self.assertIsInstance(tokenizer, BytePairTokenizer)

    def test_build_text_tokenizer_can_build_byte_baseline(self) -> None:
        tokenizer = build_text_tokenizer("hello", kind="byte")

        self.assertIsInstance(tokenizer, ByteTokenizer)

    def test_byte_pair_tokenizer_round_trips_unseen_text(self) -> None:
        tokenizer = train_byte_pair_tokenizer("hello hello 箱 hello", vocab_size=270)

        token_ids = tokenizer.encode("hello 箱 unseen")

        self.assertIsInstance(tokenizer, BytePairTokenizer)
        self.assertEqual(tokenizer.decode(token_ids), "hello 箱 unseen")
        self.assertGreaterEqual(tokenizer.vocab_size, 267)
        self.assertLessEqual(tokenizer.vocab_size, 270)

    def test_serializes_byte_tokenizer(self) -> None:
        tokenizer = text_tokenizer_from_payload(text_tokenizer_to_payload(ByteTokenizer()))

        self.assertIsInstance(tokenizer, ByteTokenizer)
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_saves_and_loads_byte_pair_text_tokenizer_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            original = train_byte_pair_tokenizer("hello hello", vocab_size=270)

            save_text_tokenizer(path, original)
            restored = load_text_tokenizer(path)

        self.assertIsInstance(restored, BytePairTokenizer)
        self.assertEqual(restored.decode(restored.encode("hello unseen")), "hello unseen")


if __name__ == "__main__":
    unittest.main()
