from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.corpus_coverage import (
    format_coverage_text,
    main,
    summarize_corpus_coverage,
)
from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


class CorpusCoverageTest(unittest.TestCase):
    def test_summarizes_modalities_markers_and_lengths(self) -> None:
        documents = [
            MixedDocument(
                id="external_action_001",
                modality="external_action",
                content="<obs> home <action> click checkout <next_obs> cart",
            ),
            MixedDocument(
                id="external_action_002",
                modality="external_action",
                content="<obs> cart <action> click pay",
            ),
            MixedDocument(id="text_001", modality="external_text", content="plain background"),
        ]

        coverage = summarize_corpus_coverage(documents)

        self.assertEqual(coverage.document_count, 3)
        self.assertEqual(
            [(item.modality, item.document_count) for item in coverage.modalities],
            [("external_action", 2), ("external_text", 1)],
        )
        self.assertEqual(
            coverage.marker_document_counts,
            {"<obs>": 2, "<action>": 2, "<next_obs>": 1},
        )
        self.assertEqual(
            coverage.length.total_bytes,
            sum(len(document.content.encode("utf-8")) for document in documents),
        )
        self.assertEqual(coverage.length.total_token_proxy, 14)

    def test_empty_corpus_reports_zero_lengths(self) -> None:
        coverage = summarize_corpus_coverage([])

        self.assertEqual(coverage.document_count, 0)
        self.assertEqual(coverage.modalities, ())
        self.assertEqual(coverage.length.total_bytes, 0)
        self.assertEqual(coverage.length.mean_bytes, 0.0)
        self.assertEqual(
            coverage.marker_document_counts,
            {"<obs>": 0, "<action>": 0, "<next_obs>": 0},
        )

    def test_formats_text_report_without_taxonomy(self) -> None:
        coverage = summarize_corpus_coverage(
            [
                MixedDocument(
                    id="doc",
                    modality="external_action",
                    content="<obs> a <action> b <next_obs> c",
                )
            ]
        )

        report = format_coverage_text(coverage)

        self.assertIn("documents\t1", report)
        self.assertIn("markers\t<obs>=1\t<action>=1\t<next_obs>=1", report)
        self.assertIn("external_action\tdocuments=1", report)

    def test_cli_reports_json_from_mixed_document_jsonl(self) -> None:
        output = io.StringIO()
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "mixed.jsonl"
            write_mixed_documents_jsonl(
                input_path,
                [
                    MixedDocument(
                        id="doc",
                        modality="external_action",
                        content="<obs> home <action> click",
                    )
                ],
            )

            with redirect_stdout(output):
                main(["--input", str(input_path), "--format", "json"])

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["marker_document_counts"]["<next_obs>"], 0)
        self.assertEqual(payload["modalities"][0]["modality"], "external_action")

    def test_cli_reports_input_errors(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(["--input", "missing.jsonl"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("missing.jsonl", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
