from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


@dataclass(frozen=True)
class SourceCandidate:
    id: str
    name: str
    homepage_url: str
    license_hint: str
    adapter: str
    notes: str


TEXT_JSONL_ADAPTER = "text-jsonl"


def curated_source_candidates() -> list[SourceCandidate]:
    return [
        SourceCandidate(
            id="project_gutenberg",
            name="Project Gutenberg",
            homepage_url="https://www.gutenberg.org/",
            license_hint="Public domain in the United States for many works; verify per item.",
            adapter=TEXT_JSONL_ADAPTER,
            notes="Long-form public-domain book text after local download and cleanup.",
        ),
        SourceCandidate(
            id="wikimedia_dumps",
            name="Wikimedia dumps",
            homepage_url="https://dumps.wikimedia.org/",
            license_hint="Varies by project; commonly CC BY-SA or GFDL.",
            adapter=TEXT_JSONL_ADAPTER,
            notes="Article text exported locally to JSONL with one text field per record.",
        ),
        SourceCandidate(
            id="stack_exchange_archive",
            name="Stack Exchange Data Dump",
            homepage_url="https://archive.org/details/stackexchange",
            license_hint="CC BY-SA; verify version-specific attribution requirements.",
            adapter=TEXT_JSONL_ADAPTER,
            notes="Question and answer text after local XML extraction to JSONL.",
        ),
        SourceCandidate(
            id="openwebtext_exports",
            name="OpenWebText-style local exports",
            homepage_url="https://skylion007.github.io/OpenWebTextCorpus/",
            license_hint="Derived web text; verify source and redistribution constraints.",
            adapter=TEXT_JSONL_ADAPTER,
            notes="Locally prepared web text records. This command does not download data.",
        ),
    ]


def convert_text_jsonl_to_mixed_documents(
    input_path: str | Path,
    *,
    source_id: str,
    text_field: str = "text",
    id_field: str | None = "id",
    modality: str = "external_text",
    limit: int | None = None,
) -> list[MixedDocument]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source_component = _safe_identifier_component(source_id)
    documents: list[MixedDocument] = []
    lines = Path(input_path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if limit is not None and len(documents) >= limit:
            break
        if not line.strip():
            continue
        record = _load_jsonl_object(line, line_number)
        if text_field not in record:
            raise ValueError(f"Invalid JSONL record at line {line_number}: missing text field {text_field}")
        text = record[text_field]
        if not isinstance(text, str):
            raise ValueError(f"Invalid JSONL record at line {line_number}: field {text_field} must be a string")
        document_id = _mixed_document_id(
            source_component,
            record,
            line_number=line_number,
            id_field=id_field,
        )
        documents.append(MixedDocument(id=document_id, modality=modality, content=text))
    return documents


def write_converted_text_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_id: str,
    text_field: str = "text",
    id_field: str | None = "id",
    modality: str = "external_text",
    limit: int | None = None,
) -> list[MixedDocument]:
    documents = convert_text_jsonl_to_mixed_documents(
        input_path,
        source_id=source_id,
        text_field=text_field,
        id_field=id_field,
        modality=modality,
        limit=limit,
    )
    write_mixed_documents_jsonl(output_path, documents)
    return documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List external corpus candidates or adapt local JSONL into MixedDocument JSONL."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List curated public dataset candidates.")
    list_parser.add_argument(
        "--format",
        choices=("text", "jsonl"),
        default="text",
        help="Output candidate manifest as compact text or JSONL.",
    )

    convert_parser = subparsers.add_parser(
        "convert-jsonl",
        help="Convert a local JSONL text export into MixedDocument JSONL.",
    )
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument("--output", type=Path, required=True)
    convert_parser.add_argument("--source-id", required=True)
    convert_parser.add_argument("--adapter", choices=(TEXT_JSONL_ADAPTER,), default=TEXT_JSONL_ADAPTER)
    convert_parser.add_argument("--text-field", default="text")
    convert_parser.add_argument("--id-field", default="id")
    convert_parser.add_argument("--no-id-field", action="store_true")
    convert_parser.add_argument("--modality", default="external_text")
    convert_parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        _print_candidates(args.format)
        return

    if args.command == "convert-jsonl":
        id_field = None if args.no_id_field else args.id_field
        try:
            documents = write_converted_text_jsonl(
                args.input,
                args.output,
                source_id=args.source_id,
                text_field=args.text_field,
                id_field=id_field,
                modality=args.modality,
                limit=args.limit,
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        print(f"wrote {len(documents)} mixed documents to {args.output}")
        return

    raise AssertionError(f"unsupported command: {args.command}")


def _print_candidates(output_format: str) -> None:
    candidates = curated_source_candidates()
    if output_format == "jsonl":
        for candidate in candidates:
            print(
                json.dumps(
                    {
                        "id": candidate.id,
                        "name": candidate.name,
                        "homepage_url": candidate.homepage_url,
                        "license_hint": candidate.license_hint,
                        "adapter": candidate.adapter,
                        "notes": candidate.notes,
                    },
                    ensure_ascii=False,
                )
            )
        return
    for candidate in candidates:
        print(
            f"{candidate.id}\t{candidate.name}\tadapter={candidate.adapter}"
            f"\tlicense={candidate.license_hint}"
        )


def _load_jsonl_object(line: str, line_number: int) -> dict[str, object]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
    if not isinstance(record, dict):
        raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
    return record


def _mixed_document_id(
    source_component: str,
    record: dict[str, object],
    *,
    line_number: int,
    id_field: str | None,
) -> str:
    if id_field is None or id_field not in record:
        return f"{source_component}_{line_number:06d}"
    raw_id = record[id_field]
    if not isinstance(raw_id, str | int):
        raise ValueError(f"Invalid JSONL record at line {line_number}: field {id_field} must be a string or int")
    return f"{source_component}_{_safe_identifier_component(str(raw_id))}"


def _safe_identifier_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    if not component:
        raise ValueError("identifier components must contain at least one safe character")
    return component


if __name__ == "__main__":
    main()
