import unittest

from intrep.text_examples import LanguageModelingExample, language_modeling_corpus_from_examples


class LanguageModelingExampleTest(unittest.TestCase):
    def test_language_modeling_example_rejects_empty_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            LanguageModelingExample("")

    def test_language_modeling_corpus_from_examples_joins_examples_with_newlines(self) -> None:
        corpus = language_modeling_corpus_from_examples(
            (
                LanguageModelingExample("alpha beta"),
                LanguageModelingExample("gamma delta"),
            )
        )

        self.assertEqual(corpus, "alpha beta\ngamma delta")

    def test_language_modeling_corpus_from_examples_rejects_empty_examples(self) -> None:
        with self.assertRaisesRegex(ValueError, "examples must not be empty"):
            language_modeling_corpus_from_examples(())


if __name__ == "__main__":
    unittest.main()
