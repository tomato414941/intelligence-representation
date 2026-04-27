import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import (
    MixedDocument,
    default_mixed_documents,
    load_mixed_documents_jsonl,
    render_corpus,
    write_mixed_documents_jsonl,
)


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

    def test_load_mixed_documents_jsonl_reads_records(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                '{"id":"text_001","modality":"text","content":"hello"}\n'
                '{"id":"code_001","modality":"code","content":"print(1)"}\n',
                encoding="utf-8",
            )

            documents = load_mixed_documents_jsonl(path)

        self.assertEqual(
            documents,
            [
                MixedDocument(id="text_001", modality="text", content="hello"),
                MixedDocument(id="code_001", modality="code", content="print(1)"),
            ],
        )

    def test_load_mixed_documents_jsonl_validates_missing_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text('{"id":"broken","content":"hello"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: missing required fields: modality"):
                load_mixed_documents_jsonl(path)

    def test_load_mixed_documents_jsonl_validates_field_types(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text('{"id":"broken","modality":"text","content":["hello"]}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: field content must be a string"):
                load_mixed_documents_jsonl(path)

    def test_render_corpus_rejects_broken_document_boundaries(self) -> None:
        documents = [MixedDocument(id="bad id", modality="text", content="hello")]

        with self.assertRaisesRegex(ValueError, "document id is not renderable"):
            render_corpus(documents)

    def test_write_mixed_documents_jsonl_round_trips_documents(self) -> None:
        documents = [
            MixedDocument(id="ja_001", modality="text", content="鍵は箱の中にある。"),
            MixedDocument(id="env_001", modality="environment_symbolic", content="<obs> box closed"),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"

            write_mixed_documents_jsonl(path, documents)

            self.assertEqual(load_mixed_documents_jsonl(path), documents)
            self.assertIn("鍵は箱の中にある", render_corpus(load_mixed_documents_jsonl(path)))


if __name__ == "__main__":
    unittest.main()
