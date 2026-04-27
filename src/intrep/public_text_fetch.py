from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen

from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


Downloader = Callable[[str], str | bytes]


def fetch_public_text_document(
    url: str,
    *,
    downloader: Downloader | None = None,
    source_id: str = "project_gutenberg",
    modality: str = "external_text",
    document_id: str | None = None,
    index: int = 1,
) -> MixedDocument:
    _validate_http_url(url)
    read = downloader or download_text_url
    text = strip_gutenberg_boilerplate(_decode_text(read(url)))
    if not text:
        raise ValueError(f"downloaded text is empty after cleanup: {url}")
    return MixedDocument(
        id=document_id or _document_id_from_url(source_id, url, index),
        modality=modality,
        content=text,
    )


def fetch_public_text_documents(
    urls: Iterable[str],
    *,
    downloader: Downloader | None = None,
    source_id: str = "project_gutenberg",
    modality: str = "external_text",
) -> list[MixedDocument]:
    url_list = list(urls)
    for url in url_list:
        _validate_http_url(url)

    documents: list[MixedDocument] = []
    seen_ids: set[str] = set()
    for index, url in enumerate(url_list, start=1):
        document = fetch_public_text_document(
            url,
            downloader=downloader,
            source_id=source_id,
            modality=modality,
            index=index,
        )
        if document.id in seen_ids:
            document = MixedDocument(
                id=f"{document.id}_{index:06d}",
                modality=document.modality,
                content=document.content,
            )
        seen_ids.add(document.id)
        documents.append(document)
    return documents


def write_fetched_public_text_jsonl(
    urls: Iterable[str],
    output_path: str | Path,
    *,
    downloader: Downloader | None = None,
    source_id: str = "project_gutenberg",
    modality: str = "external_text",
) -> list[MixedDocument]:
    documents = fetch_public_text_documents(
        urls,
        downloader=downloader,
        source_id=source_id,
        modality=modality,
    )
    write_mixed_documents_jsonl(output_path, documents)
    return documents


def download_text_url(url: str) -> str:
    _validate_http_url(url)
    with urlopen(url, timeout=30) as response:
        payload = response.read()
        charset = response.headers.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def strip_gutenberg_boilerplate(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = normalized.split("\n")

    start_index = 0
    for index, line in enumerate(lines):
        if _is_gutenberg_start_marker(line):
            start_index = index + 1
            break

    body_lines = lines[start_index:]
    end_index = len(body_lines)
    for index, line in enumerate(body_lines):
        if _is_gutenberg_end_marker(line):
            end_index = index
            break

    return "\n".join(body_lines[:end_index]).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch public-domain plain text URLs into MixedDocument JSONL."
    )
    parser.add_argument(
        "--url",
        action="append",
        required=True,
        help="HTTP(S) plain text URL. Repeat to fetch multiple texts.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", default="project_gutenberg")
    parser.add_argument("--modality", default="external_text")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        documents = write_fetched_public_text_jsonl(
            args.url,
            args.output,
            source_id=args.source_id,
            modality=args.modality,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {len(documents)} mixed documents to {args.output}")


def _validate_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"only http(s) URLs are supported: {url}")


def _decode_text(payload: str | bytes) -> str:
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8", errors="replace")


def _is_gutenberg_start_marker(line: str) -> bool:
    normalized = _normalized_marker_line(line)
    return (
        normalized.startswith("*** START OF ")
        and "PROJECT GUTENBERG" in normalized
        and "EBOOK" in normalized
    )


def _is_gutenberg_end_marker(line: str) -> bool:
    normalized = _normalized_marker_line(line)
    return (
        (
            normalized.startswith("*** END OF ")
            or normalized.startswith("END OF THE PROJECT GUTENBERG")
            or normalized.startswith("END OF PROJECT GUTENBERG")
        )
        and "PROJECT GUTENBERG" in normalized
        and "EBOOK" in normalized
    )


def _normalized_marker_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().upper())


def _document_id_from_url(source_id: str, url: str, index: int) -> str:
    source_component = _safe_identifier_component(source_id)
    parsed = urlparse(url)
    path = parsed.path
    gutenberg_id = _gutenberg_numeric_id(path)
    if gutenberg_id is not None:
        return f"{source_component}_{gutenberg_id}"

    name = Path(path).stem
    if name:
        return f"{source_component}_{_safe_identifier_component(name)}"
    return f"{source_component}_{index:06d}"


def _gutenberg_numeric_id(path: str) -> str | None:
    match = re.search(r"/ebooks/(\d+)(?:\D|$)", path)
    if match:
        return match.group(1)
    match = re.search(r"/files/(\d+)(?:/|$)", path)
    if match:
        return match.group(1)
    match = re.search(r"/epub/(\d+)(?:/|$)", path)
    if match:
        return match.group(1)
    match = re.search(r"/(?:pg)?(\d+)(?:-\d+)?\.txt$", path)
    if match:
        return match.group(1)
    return None


def _safe_identifier_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    if not component:
        raise ValueError("identifier components must contain at least one safe character")
    return component


if __name__ == "__main__":
    main()
