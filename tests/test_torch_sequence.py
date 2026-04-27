import unittest

from intrep.sequence import SequenceExample
from intrep.torch_sequence import PAD_TOKEN, build_vocabulary


class TorchSequenceTest(unittest.TestCase):
    def test_encode_tokens_left_pads_short_inputs(self) -> None:
        vocabulary = build_vocabulary(
            [
                SequenceExample(
                    id="short",
                    input_tokens=["ACTION:find:財布:unknown", "PREDICT"],
                    target_token="FACT:財布:located_at:机",
                    source="test",
                )
            ]
        )

        encoded = vocabulary.encode_tokens(["ACTION:find:財布:unknown", "PREDICT"], 4)

        self.assertEqual(encoded[:2], [vocabulary.token_to_id[PAD_TOKEN]] * 2)
        self.assertEqual(
            encoded[2:],
            [
                vocabulary.token_to_id["ACTION:find:財布:unknown"],
                vocabulary.token_to_id["PREDICT"],
            ],
        )

    def test_encode_tokens_truncates_to_suffix(self) -> None:
        vocabulary = build_vocabulary(
            [
                SequenceExample(
                    id="long",
                    input_tokens=["FACT:a", "FACT:b", "ACTION:find:財布:unknown", "PREDICT"],
                    target_token="FACT:財布:located_at:机",
                    source="test",
                )
            ]
        )

        encoded = vocabulary.encode_tokens(
            ["FACT:a", "FACT:b", "ACTION:find:財布:unknown", "PREDICT"],
            2,
        )

        self.assertEqual(
            encoded,
            [
                vocabulary.token_to_id["ACTION:find:財布:unknown"],
                vocabulary.token_to_id["PREDICT"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
