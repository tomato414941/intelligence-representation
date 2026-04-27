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
QA_JSONL_ADAPTER = "qa-jsonl"
DIALOGUE_JSONL_ADAPTER = "dialogue-jsonl"
INSTRUCTION_JSONL_ADAPTER = "instruction-jsonl"
JSONL_ADAPTERS = (
    TEXT_JSONL_ADAPTER,
    QA_JSONL_ADAPTER,
    DIALOGUE_JSONL_ADAPTER,
    INSTRUCTION_JSONL_ADAPTER,
)


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
            adapter=QA_JSONL_ADAPTER,
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
    return convert_jsonl_to_mixed_documents(
        input_path,
        source_id=source_id,
        adapter=TEXT_JSONL_ADAPTER,
        text_field=text_field,
        id_field=id_field,
        modality=modality,
        limit=limit,
    )


def convert_jsonl_to_mixed_documents(
    input_path: str | Path,
    *,
    source_id: str,
    adapter: str = TEXT_JSONL_ADAPTER,
    text_field: str = "text",
    id_field: str | None = "id",
    modality: str = "external_text",
    limit: int | None = None,
) -> list[MixedDocument]:
    if adapter not in JSONL_ADAPTERS:
        raise ValueError(f"unsupported JSONL adapter: {adapter}")
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
        text = _record_content(record, adapter=adapter, text_field=text_field, line_number=line_number)
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
    documents = write_converted_jsonl(
        input_path,
        output_path,
        source_id=source_id,
        adapter=TEXT_JSONL_ADAPTER,
        text_field=text_field,
        id_field=id_field,
        modality=modality,
        limit=limit,
    )
    return documents


def write_converted_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_id: str,
    adapter: str = TEXT_JSONL_ADAPTER,
    text_field: str = "text",
    id_field: str | None = "id",
    modality: str = "external_text",
    limit: int | None = None,
) -> list[MixedDocument]:
    documents = convert_jsonl_to_mixed_documents(
        input_path,
        source_id=source_id,
        adapter=adapter,
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
    convert_parser.add_argument("--adapter", choices=JSONL_ADAPTERS, default=TEXT_JSONL_ADAPTER)
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
            documents = write_converted_jsonl(
                args.input,
                args.output,
                source_id=args.source_id,
                adapter=args.adapter,
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


def _record_content(
    record: dict[str, object],
    *,
    adapter: str,
    text_field: str,
    line_number: int,
) -> str:
    if adapter == TEXT_JSONL_ADAPTER:
        return _required_string_field(record, (text_field,), line_number, f"text field {text_field}")
    if adapter == QA_JSONL_ADAPTER:
        return _render_qa_record(record, line_number)
    if adapter == DIALOGUE_JSONL_ADAPTER:
        return _render_dialogue_record(record, line_number)
    if adapter == INSTRUCTION_JSONL_ADAPTER:
        return _render_instruction_record(record, line_number)
    raise ValueError(f"unsupported JSONL adapter: {adapter}")


def _render_qa_record(record: dict[str, object], line_number: int) -> str:
    question = _required_string_field(record, ("question", "query", "prompt"), line_number, "question")
    answer = _answer_text(record, line_number)
    return f"Question: {question}\nAnswer: {answer}"


def _render_dialogue_record(record: dict[str, object], line_number: int) -> str:
    field_name = "messages" if "messages" in record else "turns" if "turns" in record else None
    if field_name is None:
        raise ValueError(f"Invalid JSONL record at line {line_number}: missing messages or turns field")
    entries = record[field_name]
    if not isinstance(entries, list):
        raise ValueError(f"Invalid JSONL record at line {line_number}: field {field_name} must be a list")
    if not entries:
        raise ValueError(f"Invalid JSONL record at line {line_number}: field {field_name} must not be empty")
    lines = [_dialogue_line(entry, index, line_number) for index, entry in enumerate(entries, start=1)]
    return "\n".join(lines)


def _render_instruction_record(record: dict[str, object], line_number: int) -> str:
    instruction = _required_string_field(record, ("instruction", "prompt"), line_number, "instruction")
    output = _required_string_field(record, ("output", "response", "answer", "completion"), line_number, "output")
    input_text = _optional_string_field(record, ("input", "context"), line_number)
    lines = [f"Instruction: {instruction}"]
    if input_text is not None and input_text:
        lines.append(f"Input: {input_text}")
    lines.append(f"Output: {output}")
    return "\n".join(lines)


def _answer_text(record: dict[str, object], line_number: int) -> str:
    for field_name in ("answer", "response", "accepted_answer", "completion"):
        if field_name in record:
            return _string_value(record[field_name], line_number, field_name)
    if "answers" not in record:
        raise ValueError(f"Invalid JSONL record at line {line_number}: missing answer")
    answers = record["answers"]
    if not isinstance(answers, list):
        raise ValueError(f"Invalid JSONL record at line {line_number}: field answers must be a list")
    rendered_answers = [
        _text_from_string_or_mapping(answer, line_number, f"answers[{index}]")
        for index, answer in enumerate(answers)
    ]
    rendered_answers = [answer for answer in rendered_answers if answer]
    if not rendered_answers:
        raise ValueError(f"Invalid JSONL record at line {line_number}: field answers must contain text")
    return "\n\n".join(rendered_answers)


def _dialogue_line(entry: object, index: int, line_number: int) -> str:
    if isinstance(entry, str):
        return f"Turn {index}: {entry}"
    if not isinstance(entry, dict):
        raise ValueError(
            f"Invalid JSONL record at line {line_number}: dialogue entry {index} must be a string or object"
        )
    role = _optional_string_field(entry, ("role", "speaker", "from"), line_number) or f"Turn {index}"
    content = _required_string_field(entry, ("content", "text", "value"), line_number, f"dialogue entry {index} content")
    return f"{_role_label(role)}: {content}"


def _required_string_field(
    record: dict[str, object],
    field_names: tuple[str, ...],
    line_number: int,
    label: str,
) -> str:
    for field_name in field_names:
        if field_name in record:
            return _string_value(record[field_name], line_number, field_name)
    if len(field_names) == 1:
        raise ValueError(f"Invalid JSONL record at line {line_number}: missing {label}")
    fields = ", ".join(field_names)
    raise ValueError(f"Invalid JSONL record at line {line_number}: missing {label} field ({fields})")


def _optional_string_field(
    record: dict[str, object],
    field_names: tuple[str, ...],
    line_number: int,
) -> str | None:
    for field_name in field_names:
        if field_name in record:
            return _string_value(record[field_name], line_number, field_name)
    return None


def _text_from_string_or_mapping(value: object, line_number: int, label: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _required_string_field(value, ("text", "content", "body", "answer"), line_number, label)
    raise ValueError(f"Invalid JSONL record at line {line_number}: {label} must be a string or object")


def _string_value(value: object, line_number: int, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Invalid JSONL record at line {line_number}: field {field_name} must be a string")
    return value


def _role_label(role: str) -> str:
    normalized = re.sub(r"[_-]+", " ", role.strip())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.title() if normalized else "Turn"


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
