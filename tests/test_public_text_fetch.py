from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.public_text_fetch import (
    PublicTextFetchPolicy,
    download_text_url,
    fetch_public_text_document,
    fetch_public_text_documents,
    main,
    strip_gutenberg_boilerplate,
    write_fetched_public_text_jsonl,
)
from intrep.source_manifest import load_source_manifest_jsonl


class PublicTextFetchTests(unittest.TestCase):
    def test_strips_obvious_gutenberg_header_and_footer(self) -> None:
        raw = (
            "Project Gutenberg preamble\n"
            "license text\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***\n"
            "\n"
            "CHAPTER I\n"
            "A useful public-domain paragraph.\n"
            "\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE BOOK ***\n"
            "footer license text\n"
        )

        cleaned = strip_gutenberg_boilerplate(raw)

        self.assertEqual(cleaned, "CHAPTER I\nA useful public-domain paragraph.")

    def test_fetch_refuses_non_http_urls_before_downloading(self) -> None:
        calls: list[str] = []

        def downloader(url: str) -> str:
            calls.append(url)
            return "unused"

        with self.assertRaisesRegex(ValueError, "only http\\(s\\) URLs"):
            fetch_public_text_document("file:///tmp/book.txt", downloader=downloader)

        self.assertEqual(calls, [])

    def test_fetch_refuses_private_or_local_hosts_before_downloading(self) -> None:
        calls: list[str] = []

        def downloader(url: str) -> str:
            calls.append(url)
            return "unused"

        with self.assertRaisesRegex(ValueError, "private or local URL hosts"):
            fetch_public_text_document("http://127.0.0.1/book.txt", downloader=downloader)

        self.assertEqual(calls, [])

    def test_fetch_policy_can_restrict_hosts(self) -> None:
        calls: list[str] = []

        def downloader(url: str) -> str:
            calls.append(url)
            return "allowed text"

        policy = PublicTextFetchPolicy(allowed_hosts=("example.org",))

        document = fetch_public_text_document(
            "https://example.org/book.txt",
            downloader=downloader,
            policy=policy,
        )

        self.assertEqual(document.content, "allowed text")
        self.assertEqual(calls, ["https://example.org/book.txt"])
        with self.assertRaisesRegex(ValueError, "not allowed"):
            fetch_public_text_document(
                "https://example.net/book.txt",
                downloader=downloader,
                policy=policy,
            )

    def test_download_rejects_disallowed_content_type_before_reading_body(self) -> None:
        response = _FakeResponse(
            payload=b"unused",
            content_type="application/octet-stream",
        )

        with patch("intrep.public_text_fetch.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "unsupported content type"):
                download_text_url("https://example.org/book.bin")

        self.assertFalse(response.read_called)

    def test_download_accepts_allowed_text_content_type(self) -> None:
        response = _FakeResponse(
            payload=b"plain text",
            content_type="text/plain",
        )

        with patch("intrep.public_text_fetch.urlopen", return_value=response):
            text = download_text_url("https://example.org/book.txt")

        self.assertEqual(text, "plain text")
        self.assertTrue(response.read_called)

    def test_fetch_documents_validates_all_urls_before_downloading(self) -> None:
        calls: list[str] = []

        def downloader(url: str) -> str:
            calls.append(url)
            return "unused"

        with self.assertRaisesRegex(ValueError, "only http\\(s\\) URLs"):
            fetch_public_text_documents(
                ["https://example.org/book.txt", "file:///tmp/book.txt"],
                downloader=downloader,
            )

        self.assertEqual(calls, [])

    def test_fetch_uses_injected_downloader_and_builds_mixed_documents(self) -> None:
        payloads = {
            "https://www.gutenberg.org/files/1342/1342-0.txt": (
                "*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
                "It is a truth universally acknowledged.\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***\n"
            ),
            "https://example.org/books/plain.txt": b"Plain public-domain text.",
        }

        def downloader(url: str) -> str | bytes:
            return payloads[url]

        documents = fetch_public_text_documents(
            payloads.keys(),
            downloader=downloader,
            source_id="public text",
            modality="external_book",
        )

        self.assertEqual(
            [document.id for document in documents],
            ["public_text_1342", "public_text_plain"],
        )
        self.assertEqual(
            [document.modality for document in documents],
            ["external_book", "external_book"],
        )
        self.assertEqual(documents[0].content, "It is a truth universally acknowledged.")
        self.assertEqual(documents[1].content, "Plain public-domain text.")

    def test_write_fetched_public_text_jsonl_round_trips_as_mixed_documents(self) -> None:
        def downloader(url: str) -> str:
            return f"Text from {url}"

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "mixed.jsonl"

            documents = write_fetched_public_text_jsonl(
                ["https://www.gutenberg.org/cache/epub/84/pg84.txt"],
                output_path,
                downloader=downloader,
            )
            loaded = load_mixed_documents_jsonl(output_path)

        self.assertEqual(documents, loaded)
        self.assertEqual(loaded[0].id, "project_gutenberg_84")
        self.assertEqual(
            loaded[0].content,
            "Text from https://www.gutenberg.org/cache/epub/84/pg84.txt",
        )

    def test_write_fetched_public_text_jsonl_writes_manifest_sidecar(self) -> None:
        def downloader(url: str) -> str:
            return f"Text from {url}"

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "mixed.jsonl"
            manifest_path = Path(directory) / "manifest.jsonl"

            write_fetched_public_text_jsonl(
                ["https://example.org/books/plain.txt"],
                output_path,
                downloader=downloader,
                source_id="public_text",
                manifest_path=manifest_path,
            )
            manifest = load_source_manifest_jsonl(manifest_path)

        self.assertEqual(len(manifest), 1)
        self.assertEqual(manifest[0].document_id, "public_text_plain")
        self.assertEqual(manifest[0].source_url, "https://example.org/books/plain.txt")
        self.assertEqual(manifest[0].adapter, "public-text-url")

    def test_cli_reports_non_http_url_without_fetching_data(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(["--url", "ftp://example.org/book.txt", "--output", "out.jsonl"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("only http(s) URLs are supported", error_output.getvalue())


class _FakeResponse:
    def __init__(self, *, payload: bytes, content_type: str) -> None:
        self.payload = payload
        self.read_called = False
        self.headers = Message()
        self.headers["Content-Type"] = content_type

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_called = True
        if size < 0:
            return self.payload
        return self.payload[:size]


if __name__ == "__main__":
    unittest.main()
