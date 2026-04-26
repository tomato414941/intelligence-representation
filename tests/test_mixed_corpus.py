import unittest

from intrep.mixed_corpus import default_mixed_documents, render_corpus


class MixedCorpusTest(unittest.TestCase):
    def test_default_corpus_contains_expected_modalities(self) -> None:
        documents = default_mixed_documents()

        modalities = {document.modality for document in documents}

        self.assertIn("text", modalities)
        self.assertIn("environment_symbolic", modalities)
        self.assertIn("environment_natural", modalities)
        self.assertIn("code", modalities)
        self.assertIn("log", modalities)

    def test_render_corpus_uses_lightweight_tags(self) -> None:
        rendered = render_corpus(default_mixed_documents())

        self.assertIn("<doc type=text", rendered)
        self.assertIn("<obs>", rendered)
        self.assertIn("<action>", rendered)
        self.assertIn("<next_obs>", rendered)
        self.assertIn("鍵は箱の中にある", rendered)


if __name__ == "__main__":
    unittest.main()
