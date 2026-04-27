from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.source_manifest import (
    TEXT_JSONL_ADAPTER,
    convert_text_jsonl_to_mixed_documents,
    curated_source_candidates,
    main,
)


class SourceManifestTest(unittest.TestCase):
    def test_curated_candidates_are_thin_public_source_manifest(self) -> None:
        candidates = curated_source_candidates()

        self.assertGreaterEqual(len(candidates), 3)
        self.assertIn("project_gutenberg", {candidate.id for candidate in candidates})
        for candidate in candidates:
            self.assertTrue(candidate.id)
            self.assertTrue(candidate.name)
            self.assertTrue(candidate.homepage_url.startswith("https://"))
            self.assertTrue(candidate.license_hint)
            self.assertEqual(candidate.adapter, TEXT_JSONL_ADAPTER)

    def test_list_command_prints_candidate_manifest(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main(["list"])

        stdout = output.getvalue()
        self.assertIn("project_gutenberg\tProject Gutenberg\tadapter=text-jsonl", stdout)
        self.assertIn("wikimedia_dumps\tWikimedia dumps\tadapter=text-jsonl", stdout)

    def test_list_command_can_print_jsonl(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            main(["list", "--format", "jsonl"])

        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(rows[0]["id"], "project_gutenberg")
        self.assertEqual(rows[0]["adapter"], TEXT_JSONL_ADAPTER)
        self.assertIn("license_hint", rows[0])

    def test_convert_text_jsonl_to_mixed_documents_uses_local_records(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            input_path.write_text(
                '{"id":"doc 1","text":"First document."}\n'
                '{"id":2,"text":"Second document."}\n'
                '{"text":"Third document."}\n',
                encoding="utf-8",
            )

            documents = convert_text_jsonl_to_mixed_documents(
                input_path,
                source_id="sample source",
                modality="external_book",
            )

        self.assertEqual([document.id for document in documents], ["sample-source_doc-1", "sample-source_2", "sample-source_000003"])
        self.assertEqual([document.modality for document in documents], ["external_book"] * 3)
        self.assertEqual([document.content for document in documents], ["First document.", "Second document.", "Third document."])

    def test_convert_text_jsonl_supports_custom_text_field_limit_and_no_id_field(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            input_path.write_text(
                '{"id":"ignored","body":"First"}\n'
                '{"id":"ignored-too","body":"Second"}\n',
                encoding="utf-8",
            )

            documents = convert_text_jsonl_to_mixed_documents(
                input_path,
                source_id="source",
                text_field="body",
                id_field=None,
                limit=1,
            )

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].id, "source_000001")
        self.assertEqual(documents[0].content, "First")

    def test_convert_text_jsonl_validates_record_shape(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            input_path.write_text('{"id":"bad","text":["not text"]}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1: field text must be a string"):
                convert_text_jsonl_to_mixed_documents(input_path, source_id="source")

    def test_convert_command_writes_mixed_document_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            temp_path = Path(directory)
            input_path = temp_path / "input.jsonl"
            output_path = temp_path / "mixed.jsonl"
            input_path.write_text('{"id":"a","text":"Alpha"}\n{"text":"Beta"}\n', encoding="utf-8")

            with redirect_stdout(output):
                main(
                    [
                        "convert-jsonl",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--source-id",
                        "local",
                    ]
                )

            documents = load_mixed_documents_jsonl(output_path)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].id, "local_a")
        self.assertEqual(documents[0].modality, "external_text")
        self.assertEqual(documents[1].id, "local_000002")
        self.assertIn("wrote 2 mixed documents", output.getvalue())

    def test_convert_command_reports_local_input_errors(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "convert-jsonl",
                        "--input",
                        "missing.jsonl",
                        "--output",
                        "out.jsonl",
                        "--source-id",
                        "local",
                    ]
                )

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("missing.jsonl", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
