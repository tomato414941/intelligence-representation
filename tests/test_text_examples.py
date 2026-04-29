import unittest

from intrep.text_examples import RawTextExample, text_corpus_from_examples


class RawTextExampleTest(unittest.TestCase):
    def test_raw_text_example_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            RawTextExample("")

    def test_text_corpus_from_examples_joins_examples_with_newlines(self) -> None:
        corpus = text_corpus_from_examples(
            (
                RawTextExample("alpha beta"),
                RawTextExample("gamma delta"),
            )
        )

        self.assertEqual(corpus, "alpha beta\ngamma delta")

    def test_text_corpus_from_examples_rejects_empty_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            text_corpus_from_examples(())


if __name__ == "__main__":
    unittest.main()
