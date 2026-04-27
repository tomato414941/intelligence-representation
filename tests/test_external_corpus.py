from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from intrep.external_corpus import (
    adapt_external_action_record,
    adapt_external_action_records,
    list_public_data_sources,
    load_external_action_jsonl,
    main,
    write_external_action_jsonl,
)
from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.next_observation_cases import extract_next_observation_cases
from intrep.source_manifest import load_source_manifest_jsonl


class ExternalCorpusTests(unittest.TestCase):
    def test_lists_public_sources_with_web_action_candidates(self) -> None:
        sources = list_public_data_sources()

        names = {source.name for source in sources}
        self.assertIn("mind2web", names)
        self.assertIn("weblinx", names)
        self.assertTrue(all(source.url.startswith("https://") for source in sources))

    def test_adapts_generic_web_action_record_to_mixed_document(self) -> None:
        document = adapt_external_action_record(
            {
                "annotation_id": "task 1",
                "confirmed_task": "Find the checkout button.",
                "cleaned_html": "<button>Checkout</button>",
                "operation": {"op": "CLICK", "value": "Checkout"},
                "next_observation": "The cart page is open.",
            },
            source_name="mind2web",
            fallback_id="fallback",
        )

        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.modality, "external_action")
        self.assertEqual(document.id, "mind2web_task_1")
        self.assertIn("<task> Find the checkout button.", document.content)
        self.assertIn("<obs> <button>Checkout</button>", document.content)
        self.assertIn("<action> CLICK Checkout", document.content)
        self.assertIn("<next_obs> The cart page is open.", document.content)

    def test_external_action_documents_become_next_observation_cases(self) -> None:
        documents = adapt_external_action_records(
            [
                {
                    "id": "step-1",
                    "instruction": "Open the drawer.",
                    "observation": "drawer closed",
                    "action": "open drawer",
                    "next_observation": "drawer open",
                }
            ],
            source_name="sample",
        )

        cases = extract_next_observation_cases(documents)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].modality, "external_action")
        self.assertIn("<action> open drawer <next_obs>", cases[0].prefix)
        self.assertEqual(cases[0].positive_next, "drawer open")

    def test_loads_external_action_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                '{"id":"a","task":"Find help","dom":"<a>Help</a>",'
                '"operation":{"op":"CLICK","value":"Help"},"result":"help page"}\n',
                encoding="utf-8",
            )

            documents = load_external_action_jsonl(path, source_name="web")

        self.assertEqual(len(documents), 1)
        self.assertIn("<next_obs> help page", documents[0].content)

    def test_writes_external_action_manifest_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.jsonl"
            output_path = Path(directory) / "mixed.jsonl"
            manifest_path = Path(directory) / "manifest.jsonl"
            input_path.write_text(
                '{"id":"a","task":"Find help","dom":"<a>Help</a>",'
                '"operation":{"op":"CLICK","value":"Help"},"result":"help page"}\n',
                encoding="utf-8",
            )

            documents = write_external_action_jsonl(
                input_path,
                output_path,
                source_name="web",
                manifest_path=manifest_path,
            )
            manifest = load_source_manifest_jsonl(manifest_path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(manifest[0].document_id, "web_a")
        self.assertEqual(manifest[0].source_id, "web")
        self.assertEqual(manifest[0].adapter, "generic_web_navigation")
        self.assertEqual(manifest[0].input_path, str(input_path))

    def test_cli_lists_sources_and_converts_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.jsonl"
            output_path = Path(directory) / "mixed.jsonl"
            input_path.write_text(
                '{"id":"a","instruction":"Look","observation":"home",'
                '"action":"click menu","next_observation":"menu open"}\n',
                encoding="utf-8",
            )

            list_output = io.StringIO()
            with redirect_stdout(list_output):
                main(["list-sources"])
            self.assertIn("mind2web", list_output.getvalue())

            convert_output = io.StringIO()
            with redirect_stdout(convert_output):
                main(
                    [
                        "convert-jsonl",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--source-name",
                        "web",
                    ]
                )

            self.assertIn("wrote 1 mixed documents", convert_output.getvalue())
            documents = load_mixed_documents_jsonl(output_path)
            self.assertEqual(documents[0].modality, "external_action")


if __name__ == "__main__":
    unittest.main()
