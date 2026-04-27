from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory

from intrep.mixed_corpus import load_mixed_documents_jsonl
from intrep.public_text_fetch import (
    fetch_public_text_document,
    fetch_public_text_documents,
    main,
    strip_gutenberg_boilerplate,
    write_fetched_public_text_jsonl,
)


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

    def test_cli_reports_non_http_url_without_fetching_data(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as raised:
                main(["--url", "ftp://example.org/book.txt", "--output", "out.jsonl"])

        self.assertNotEqual(raised.exception.code, 0)
        self.assertIn("only http(s) URLs are supported", error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
