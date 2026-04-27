from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re

from intrep.mixed_corpus import MixedDocument, write_mixed_documents_jsonl


ExternalRecord = Mapping[str, object]
ExternalCorpusAdapter = Callable[[Iterable[ExternalRecord], str], list[MixedDocument]]


@dataclass(frozen=True)
class ExternalCorpusSource:
    name: str
    description: str
    adapter: ExternalCorpusAdapter
    homepage: str | None = None
    citation: str | None = None


@dataclass(frozen=True)
class ExternalDataSource:
    name: str
    url: str
    license: str
    modality: str
    adapter: str
    notes: str


PUBLIC_DATA_SOURCES: tuple[ExternalDataSource, ...] = (
    ExternalDataSource(
        name="mind2web",
        url="https://huggingface.co/datasets/osunlp/Mind2Web",
        license="CC-BY-4.0",
        modality="web_action",
        adapter="generic_web_navigation",
        notes="Web tasks with language instruction, HTML observations, and action traces.",
    ),
    ExternalDataSource(
        name="multimodal-mind2web",
        url="https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web",
        license="CC-BY-4.0",
        modality="web_action",
        adapter="generic_web_navigation",
        notes="Mind2Web action records with screenshots and HTML fields.",
    ),
    ExternalDataSource(
        name="weblinx",
        url="https://huggingface.co/datasets/McGill-NLP/WebLINX",
        license="CC-BY-NC-SA-4.0",
        modality="web_dialogue_action",
        adapter="generic_web_navigation",
        notes="Real-world website navigation with multi-turn dialogue and demonstrations.",
    ),
    ExternalDataSource(
        name="miniwob-plusplus",
        url="https://github.com/Farama-Foundation/miniwob-plusplus",
        license="MIT",
        modality="web_env",
        adapter="generic_web_navigation",
        notes="Interactive web environments; export local rollouts before conversion.",
    ),
    ExternalDataSource(
        name="alfworld",
        url="https://alfworld.github.io/",
        license="See upstream dataset and code licenses",
        modality="text_embodied_env",
        adapter="generic_web_navigation",
        notes="Text/embodied environment traces can be exported as observation-action records.",
    ),
)


def list_public_data_sources() -> tuple[ExternalDataSource, ...]:
    return PUBLIC_DATA_SOURCES


class ExternalCorpusRegistry:
    def __init__(self, sources: Iterable[ExternalCorpusSource] = ()) -> None:
        self._sources: dict[str, ExternalCorpusSource] = {}
        for source in sources:
            self.register(source)

    def register(self, source: ExternalCorpusSource) -> None:
        if source.name in self._sources:
            raise ValueError(f"external corpus source already registered: {source.name}")
        self._sources[source.name] = source

    def get(self, name: str) -> ExternalCorpusSource:
        try:
            return self._sources[name]
        except KeyError as error:
            raise ValueError(f"unknown external corpus source: {name}") from error

    def names(self) -> list[str]:
        return sorted(self._sources)

    def adapt(self, source_name: str, records: Iterable[ExternalRecord]) -> list[MixedDocument]:
        source = self.get(source_name)
        return source.adapter(records, source.name)


def default_external_corpus_registry() -> ExternalCorpusRegistry:
    return ExternalCorpusRegistry(
        [
            ExternalCorpusSource(
                name="generic_web_navigation",
                description=(
                    "Generic web/UI/navigation records with instruction, observation, "
                    "DOM or screenshot alt text, action, and next observation fields."
                ),
                adapter=adapt_web_navigation_records,
            )
        ]
    )


def load_external_corpus_jsonl(
    path: str | Path,
    source_name: str,
    *,
    registry: ExternalCorpusRegistry | None = None,
) -> list[MixedDocument]:
    records: list[ExternalRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
        records.append(record)
    if registry is None:
        return adapt_web_navigation_records(records, source_name)
    return registry.adapt(source_name, records)


def load_external_action_jsonl(
    path: str | Path,
    *,
    source_name: str = "generic_web_navigation",
) -> list[MixedDocument]:
    records = _read_external_records(path)
    return adapt_web_navigation_records(records, source_name)


def write_external_action_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_name: str = "generic_web_navigation",
    manifest_path: str | Path | None = None,
) -> list[MixedDocument]:
    documents = load_external_action_jsonl(input_path, source_name=source_name)
    write_mixed_documents_jsonl(output_path, documents)
    if manifest_path is not None:
        from intrep.source_manifest import SourceManifestRecord, write_source_manifest_jsonl

        source_line_numbers = [
            line_number
            for line_number, line in enumerate(
                Path(input_path).read_text(encoding="utf-8").splitlines(),
                start=1,
            )
            if line.strip()
        ]
        write_source_manifest_jsonl(
            manifest_path,
            [
                SourceManifestRecord(
                    document_id=document.id,
                    source_id=source_name,
                    source_url="",
                    license_hint="",
                    adapter="generic_web_navigation",
                    modality=document.modality,
                    input_path=str(input_path),
                    line_number=(
                        source_line_numbers[index - 1]
                        if index <= len(source_line_numbers)
                        else index
                    ),
                )
                for index, document in enumerate(documents, start=1)
            ],
        )
    return documents


def adapt_external_action_records(
    records: Iterable[ExternalRecord],
    *,
    source_name: str = "generic_web_navigation",
) -> list[MixedDocument]:
    return adapt_web_navigation_records(records, source_name)


def adapt_external_action_record(
    record: ExternalRecord,
    *,
    source_name: str = "generic_web_navigation",
    fallback_id: str,
) -> MixedDocument | None:
    documents = adapt_web_navigation_records([{**record, "_fallback_id": fallback_id}], source_name)
    return documents[0] if documents else None


def _read_external_records(path: str | Path) -> list[ExternalRecord]:
    records: list[ExternalRecord] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Invalid JSONL record at line {line_number}: expected object")
        records.append(record)
    return records


def adapt_web_navigation_records(
    records: Iterable[ExternalRecord],
    source_name: str = "generic_web_navigation",
) -> list[MixedDocument]:
    documents: list[MixedDocument] = []
    source_prefix = _renderable_id(source_name)
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"external record {index} must be an object")
        record_id = _record_id(record, index)
        document_id = f"{source_prefix}_{record_id}"
        documents.append(
            MixedDocument(
                id=document_id,
                modality="external_action",
                content=_render_web_navigation_content(record, index),
            )
        )
    return documents


def _render_web_navigation_content(record: ExternalRecord, index: int) -> str:
    instruction = _optional_text(
        record,
        (
            "instruction",
            "confirmed_task",
            "task",
            "task_name",
            "task_description",
            "goal",
            "utterance",
            "intent",
        ),
    )
    observation = _first_observation_text(
        record,
        (
            "observation",
            "cleaned_html",
            "raw_html",
            "html",
            "dom",
            "page",
            "state",
            "screenshot_alt",
        ),
        index,
        "current observation",
    )
    action = _required_action_text(record, index)
    next_observation = _first_observation_text(
        record,
        (
            "next_observation",
            "next_obs",
            "next_dom",
            "next_screenshot_alt",
            "target",
            "answer",
            "result",
            "expected_observation",
        ),
        index,
        "next observation",
    )

    lines: list[str] = []
    if instruction is not None:
        lines.append(f"<task> {instruction}")
    lines.append(f"<obs> {observation}")
    lines.append(f"<action> {action}")
    lines.append(f"<next_obs> {next_observation}")
    return " ".join(lines)


def _first_observation_text(
    record: ExternalRecord,
    field_names: tuple[str, ...],
    index: int,
    label: str,
) -> str:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return _render_value(record[field_name])
    expected = ", ".join(field_names)
    raise ValueError(f"external record {index} is missing {label}: expected one of {expected}")


def _observation_text(
    record: ExternalRecord,
    field_names: tuple[str, ...],
    index: int,
    label: str,
) -> str:
    parts: list[str] = []
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            parts.append(f"<{field_name}> {_render_value(record[field_name])}")
    if not parts:
        expected = ", ".join(field_names)
        raise ValueError(f"external record {index} is missing {label}: expected one of {expected}")
    return "\n".join(parts)


def _required_text(record: ExternalRecord, field_names: tuple[str, ...], index: int) -> str:
    text = _optional_text(record, field_names)
    if text is None:
        expected = ", ".join(field_names)
        raise ValueError(f"external record {index} is missing required field: {expected}")
    return text


def _required_action_text(record: ExternalRecord, index: int) -> str:
    action = record.get("action")
    if isinstance(action, Mapping):
        operation = action.get("operation")
        if isinstance(operation, Mapping):
            return _render_operation(operation)
        text = _optional_text(action, ("repr", "text", "action", "name", "type"))
        if text is not None:
            return text
    if isinstance(action, str):
        return action.strip()

    operation = record.get("operation")
    if isinstance(operation, Mapping):
        return _render_operation(operation)

    action_reprs = record.get("action_reprs")
    if isinstance(action_reprs, list) and action_reprs and isinstance(action_reprs[0], str):
        return action_reprs[0].strip()

    return _required_text(record, ("action_text", "action_repr", "operation_text", "op", "command"), index)


def _render_operation(operation: Mapping[str, object]) -> str:
    op = _render_value(operation.get("op") or operation.get("type") or operation.get("name"))
    value = operation.get("value")
    if value is not None:
        return f"{op} {_render_value(value)}"
    return op


def _optional_text(record: ExternalRecord, field_names: tuple[str, ...]) -> str | None:
    for field_name in field_names:
        if field_name in record and record[field_name] is not None:
            return _render_value(record[field_name])
    return None


def _record_id(record: ExternalRecord, index: int) -> str:
    value = _optional_text(
        record,
        ("id", "record_id", "episode_id", "task_id", "action_uid", "annotation_id", "_fallback_id"),
    )
    if value is None:
        return f"{index:06d}"
    return _renderable_id(value)


def _render_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _renderable_id(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    rendered = rendered.strip("_")
    if not rendered:
        raise ValueError("external record id must contain at least one renderable character")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and convert external public action-observation corpora."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-sources", help="List curated public dataset candidates.")

    convert = subparsers.add_parser(
        "convert-jsonl",
        help="Convert a local external JSONL file to MixedDocument JSONL.",
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--manifest-output", type=Path)
    convert.add_argument("--source-name", default="generic_web_navigation")
    convert.add_argument(
        "--adapter",
        choices=("generic_web_navigation",),
        default="generic_web_navigation",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-sources":
        for source in list_public_data_sources():
            print(
                f"{source.name}\t{source.license}\t{source.modality}\t"
                f"{source.adapter}\t{source.url}"
            )
        return
    if args.command == "convert-jsonl":
        documents = write_external_action_jsonl(
            args.input,
            args.output,
            source_name=args.source_name,
            manifest_path=args.manifest_output,
        )
        print(f"wrote {len(documents)} mixed documents to {args.output}")
        return
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
