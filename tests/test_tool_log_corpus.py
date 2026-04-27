from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.next_observation_cases import extract_next_observation_cases
from intrep.tool_log_corpus import (
    adapt_tool_log_record,
    adapt_tool_log_records,
    load_tool_log_jsonl,
    main,
)


class ToolLogCorpusTest(unittest.TestCase):
    def test_adapts_successful_command_record_to_marker_document(self) -> None:
        document = adapt_tool_log_record(
            {
                "id": "pytest run",
                "cwd": "/repo",
                "command": "pytest -q tests/test_example.py",
                "exit_code": 0,
                "stdout": "1 passed",
            },
            source_name="dev log",
            fallback_id="fallback",
        )

        self.assertEqual(document.id, "dev_log_pytest_run")
        self.assertEqual(document.modality, "tool_log")
        self.assertIn("<obs> cwd=/repo", document.content)
        self.assertIn("<action> pytest -q tests/test_example.py", document.content)
        self.assertIn("<next_obs> <result> exit_code=0 ; stdout=1 passed", document.content)

    def test_tool_log_documents_become_next_observation_cases(self) -> None:
        documents = adapt_tool_log_records(
            [
                {
                    "id": "build",
                    "observation": "working tree clean",
                    "cmd": "npm run build",
                    "exit_code": 0,
                    "stdout": "built",
                }
            ],
            source_name="local",
        )

        cases = extract_next_observation_cases(documents)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].id, "local_build")
        self.assertEqual(cases[0].modality, "tool_log")
        self.assertIn("<action> npm run build <next_obs> ", cases[0].prefix)
        self.assertEqual(cases[0].positive_next, "<result> exit_code=0 ; stdout=built")

    def test_adapts_failed_command_with_error_marker(self) -> None:
        document = adapt_tool_log_record(
            {
                "id": "lint",
                "argv": ["ruff", "check", "src"],
                "returncode": 1,
                "stderr": "F401 unused import",
            },
            fallback_id="fallback",
        )

        self.assertIn("<action> ruff check src", document.content)
        self.assertIn("<next_obs> <error> returncode=1 ; stderr=F401 unused import", document.content)

    def test_adapts_tool_call_record_with_explicit_next_observation(self) -> None:
        document = adapt_tool_log_record(
            {
                "tool_name": "exec_command",
                "arguments": {"cmd": "git status --short"},
                "observation": "repo before status check",
                "result": {"stdout": ""},
                "next_observation": "repo status is clean",
            },
            fallback_id="tool call 1",
        )

        self.assertEqual(document.id, "tool_log_tool_call_1")
        self.assertIn("<action> exec_command {\"cmd\": \"git status --short\"}", document.content)
        self.assertIn("<result> result={\"stdout\": \"\"}", document.content)
        self.assertTrue(document.content.endswith("<next_obs> repo status is clean"))

    def test_loads_tool_log_jsonl_and_cli_writes_mixed_documents(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "tool-log.jsonl"
            output_path = Path(directory) / "mixed.jsonl"
            input_path.write_text(
                '{"id":"one","command":"date","stdout":"Mon","exit_code":0}\n'
                '{"id":"two","tool":"search","arguments":{"q":"local"},"error":"not available"}\n',
                encoding="utf-8",
            )

            loaded = load_tool_log_jsonl(input_path, source_name="dev", limit=1)
            output = io.StringIO()
            with redirect_stdout(output):
                main(
                    [
                        "convert-jsonl",
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--source-name",
                        "dev",
                    ]
                )
            written = load_mixed_documents_jsonl(output_path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(written), 2)
        self.assertEqual(written[1].id, "dev_two")
        self.assertIn("<error> error=not available", written[1].content)
        self.assertIn("wrote 2 mixed documents", output.getvalue())

    def test_load_tool_log_jsonl_rejects_negative_limit(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "tool-log.jsonl"
            input_path.write_text('{"id":"one","command":"date"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "limit must be non-negative"):
                load_tool_log_jsonl(input_path, limit=-1)


if __name__ == "__main__":
    unittest.main()
