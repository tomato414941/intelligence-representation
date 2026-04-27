from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import ipaddress
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen

from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


Downloader = Callable[[str], str | bytes]


@dataclass(frozen=True)
class PublicTextFetchPolicy:
    allowed_schemes: tuple[str, ...] = ("http", "https")
    allowed_hosts: tuple[str, ...] | None = None
    allowed_content_types: tuple[str, ...] = ("text/*",)
    max_download_bytes: int = 10_000_000
    allow_private_hosts: bool = False


DEFAULT_PUBLIC_TEXT_FETCH_POLICY = PublicTextFetchPolicy()


def fetch_public_text_document(
    url: str,
    *,
    downloader: Downloader | None = None,
    source_id: str = "project_gutenberg",
    modality: str = "external_text",
    document_id: str | None = None,
    index: int = 1,
    policy: PublicTextFetchPolicy = DEFAULT_PUBLIC_TEXT_FETCH_POLICY,
) -> MixedDocument:
    _validate_http_url(url, policy=policy)
    read = downloader or (lambda target_url: download_text_url(target_url, policy=policy))
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
    policy: PublicTextFetchPolicy = DEFAULT_PUBLIC_TEXT_FETCH_POLICY,
) -> list[MixedDocument]:
    url_list = list(urls)
    for url in url_list:
        _validate_http_url(url, policy=policy)

    documents: list[MixedDocument] = []
    seen_ids: set[str] = set()
    for index, url in enumerate(url_list, start=1):
        document = fetch_public_text_document(
            url,
            downloader=downloader,
            source_id=source_id,
            modality=modality,
            index=index,
            policy=policy,
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
    policy: PublicTextFetchPolicy = DEFAULT_PUBLIC_TEXT_FETCH_POLICY,
    manifest_path: str | Path | None = None,
) -> list[MixedDocument]:
    url_list = list(urls)
    documents = fetch_public_text_documents(
        url_list,
        downloader=downloader,
        source_id=source_id,
        modality=modality,
        policy=policy,
    )
    write_mixed_documents_jsonl(output_path, documents)
    if manifest_path is not None:
        from intrep.source_manifest import SourceManifestRecord, write_source_manifest_jsonl

        retrieved_at = datetime.now(UTC).isoformat()
        write_source_manifest_jsonl(
            manifest_path,
            [
                SourceManifestRecord(
                    document_id=document.id,
                    source_id=source_id,
                    source_url=url,
                    license_hint="",
                    adapter="public-text-url",
                    modality=document.modality,
                    retrieved_at=retrieved_at,
                )
                for document, url in zip(documents, url_list, strict=True)
            ],
        )
    return documents


def download_text_url(
    url: str,
    *,
    policy: PublicTextFetchPolicy = DEFAULT_PUBLIC_TEXT_FETCH_POLICY,
) -> str:
    _validate_http_url(url, policy=policy)
    with urlopen(url, timeout=30) as response:
        content_type = response.headers.get_content_type()
        _validate_content_type(content_type, policy=policy)
        payload = response.read(policy.max_download_bytes + 1)
        charset = response.headers.get_content_charset() or "utf-8"
    if len(payload) > policy.max_download_bytes:
        raise ValueError(f"downloaded text exceeds maximum size: {url}")
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
    parser.add_argument("--manifest-output", type=Path)
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
            manifest_path=args.manifest_output,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {len(documents)} mixed documents to {args.output}")


def _validate_http_url(
    url: str,
    *,
    policy: PublicTextFetchPolicy = DEFAULT_PUBLIC_TEXT_FETCH_POLICY,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in policy.allowed_schemes or not parsed.netloc:
        raise ValueError(f"only http(s) URLs are supported: {url}")
    if parsed.username or parsed.password:
        raise ValueError(f"URL credentials are not supported: {url}")
    host = parsed.hostname
    if host is None:
        raise ValueError(f"URL host is required: {url}")
    normalized_host = host.lower().rstrip(".")
    if policy.allowed_hosts is not None and normalized_host not in {
        allowed.lower().rstrip(".") for allowed in policy.allowed_hosts
    }:
        raise ValueError(f"URL host is not allowed by fetch policy: {normalized_host}")
    if not policy.allow_private_hosts and _is_private_or_local_host(normalized_host):
        raise ValueError(f"private or local URL hosts are not supported: {url}")
    if policy.max_download_bytes < 1:
        raise ValueError("max_download_bytes must be positive")
    if not policy.allowed_content_types:
        raise ValueError("allowed_content_types must not be empty")


def _validate_content_type(content_type: str, *, policy: PublicTextFetchPolicy) -> None:
    normalized = content_type.lower()
    for allowed in policy.allowed_content_types:
        allowed_normalized = allowed.lower()
        if allowed_normalized.endswith("/*"):
            if normalized.startswith(allowed_normalized[:-1]):
                return
        elif normalized == allowed_normalized:
            return
    allowed_values = ", ".join(policy.allowed_content_types)
    raise ValueError(f"unsupported content type {content_type!r}; allowed: {allowed_values}")


def _decode_text(payload: str | bytes) -> str:
    if isinstance(payload, str):
        return payload
    return payload.decode("utf-8", errors="replace")


def _is_private_or_local_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"}:
        return True
    if host.endswith(".localhost") or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


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
