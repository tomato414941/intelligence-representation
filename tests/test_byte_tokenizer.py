import unittest

from intrep.byte_tokenizer import ByteTokenizer


class ByteTokenizerTest(unittest.TestCase):
    def test_round_trips_japanese_english_code_and_log_text(self) -> None:
        tokenizer = ByteTokenizer()
        text = "箱を開ける。open(box)\n[tool] status=ok"

        token_ids = tokenizer.encode(text)

        self.assertEqual(tokenizer.decode(token_ids), text)
        self.assertGreater(len(token_ids), len(text))
        self.assertEqual(tokenizer.vocab_size, 257)

    def test_decode_ignores_pad_token(self) -> None:
        tokenizer = ByteTokenizer()

        self.assertEqual(tokenizer.decode([65, tokenizer.pad_id, 66]), "AB")


if __name__ == "__main__":
    unittest.main()
