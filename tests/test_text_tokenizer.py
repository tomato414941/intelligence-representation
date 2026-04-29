import unittest

from intrep.byte_tokenizer import ByteTokenizer
from intrep.text_tokenizer import (
    BytePairTokenizer,
    build_text_tokenizer,
    train_byte_pair_tokenizer,
)


class TextTokenizerTest(unittest.TestCase):
    def test_build_text_tokenizer_keeps_byte_default(self) -> None:
        tokenizer = build_text_tokenizer("hello", kind="byte")

        self.assertIsInstance(tokenizer, ByteTokenizer)

    def test_byte_pair_tokenizer_round_trips_text(self) -> None:
        text = "hello hello 箱 hello"

        tokenizer = train_byte_pair_tokenizer(text, vocab_size=270)
        token_ids = tokenizer.encode(text)

        self.assertIsInstance(tokenizer, BytePairTokenizer)
        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertGreater(tokenizer.vocab_size, 257)
        self.assertLess(len(token_ids), len(ByteTokenizer().encode(text)))

    def test_byte_pair_tokenizer_uses_byte_fallback_for_unseen_text(self) -> None:
        tokenizer = train_byte_pair_tokenizer("aaaa bbbb", vocab_size=265)

        token_ids = tokenizer.encode("unseen text")

        self.assertEqual(tokenizer.decode(token_ids), "unseen text")

    def test_byte_pair_decode_rejects_invalid_token_id(self) -> None:
        tokenizer = train_byte_pair_tokenizer("hello hello", vocab_size=260)

        with self.assertRaisesRegex(ValueError, "invalid byte-pair token id"):
            tokenizer.decode([tokenizer.vocab_size])


if __name__ == "__main__":
    unittest.main()
