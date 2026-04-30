import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.byte_tokenizer import ByteTokenizer
from intrep.text_tokenizer import (
    BytePairTokenizer,
    SimpleBytePairTokenizer,
    build_text_tokenizer,
    load_text_tokenizer,
    save_text_tokenizer,
    text_tokenizer_from_payload,
    text_tokenizer_to_payload,
    train_byte_pair_tokenizer,
    train_simple_byte_pair_tokenizer,
)


class TextTokenizerTest(unittest.TestCase):
    def test_build_text_tokenizer_keeps_byte_default(self) -> None:
        tokenizer = build_text_tokenizer("hello", kind="byte")

        self.assertIsInstance(tokenizer, ByteTokenizer)

    def test_simple_byte_pair_tokenizer_round_trips_text(self) -> None:
        text = "hello hello 箱 hello"

        tokenizer = train_simple_byte_pair_tokenizer(text, vocab_size=270)
        token_ids = tokenizer.encode(text)

        self.assertIsInstance(tokenizer, SimpleBytePairTokenizer)
        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertGreater(tokenizer.vocab_size, 257)
        self.assertLess(len(token_ids), len(ByteTokenizer().encode(text)))

    def test_simple_byte_pair_tokenizer_uses_byte_fallback_for_unseen_text(self) -> None:
        tokenizer = train_simple_byte_pair_tokenizer("aaaa bbbb", vocab_size=265)

        token_ids = tokenizer.encode("unseen text")

        self.assertEqual(tokenizer.decode(token_ids), "unseen text")

    def test_simple_byte_pair_decode_rejects_invalid_token_id(self) -> None:
        tokenizer = train_simple_byte_pair_tokenizer("hello hello", vocab_size=260)

        with self.assertRaisesRegex(ValueError, "invalid byte-pair token id"):
            tokenizer.decode([tokenizer.vocab_size])

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

    def test_serializes_simple_byte_pair_tokenizer(self) -> None:
        original = train_simple_byte_pair_tokenizer("hello hello", vocab_size=265)
        payload = text_tokenizer_to_payload(original)

        restored = text_tokenizer_from_payload(payload)

        self.assertIsInstance(restored, SimpleBytePairTokenizer)
        self.assertEqual(restored.encode("hello hello"), original.encode("hello hello"))
        self.assertEqual(restored.decode(restored.encode("hello hello")), "hello hello")

    def test_saves_and_loads_simple_text_tokenizer_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            original = train_simple_byte_pair_tokenizer("hello hello", vocab_size=265)

            save_text_tokenizer(path, original)
            restored = load_text_tokenizer(path)

        self.assertIsInstance(restored, SimpleBytePairTokenizer)
        self.assertEqual(restored.encode("hello hello"), original.encode("hello hello"))

    def test_saves_and_loads_byte_pair_text_tokenizer_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            original = train_byte_pair_tokenizer("hello hello", vocab_size=270)

            save_text_tokenizer(path, original)
            restored = load_text_tokenizer(path)

        self.assertIsInstance(restored, BytePairTokenizer)
        self.assertEqual(restored.decode(restored.encode("hello unseen")), "hello unseen")

    def test_loads_legacy_hf_byte_pair_payload(self) -> None:
        original = train_byte_pair_tokenizer("hello hello", vocab_size=270)
        payload = text_tokenizer_to_payload(original)
        payload["kind"] = "hf-byte-pair"

        restored = text_tokenizer_from_payload(payload)

        self.assertIsInstance(restored, BytePairTokenizer)
        self.assertEqual(restored.decode(restored.encode("hello unseen")), "hello unseen")


if __name__ == "__main__":
    unittest.main()
