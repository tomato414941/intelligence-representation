from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MixedDocument:
    id: str
    modality: str
    content: str


def generate_environment_document_pairs() -> list[MixedDocument]:
    combinations = [
        ("coin", "drawer", "table"),
        ("map", "box", "desk"),
        ("token", "case", "shelf"),
    ]
    documents: list[MixedDocument] = []
    for index, (obj, container, location) in enumerate(combinations, start=1):
        suffix = f"{index:03d}"
        documents.extend(
            [
                MixedDocument(
                    id=f"env_pair_symbolic_{suffix}",
                    modality="environment_symbolic",
                    content=(
                        f"<obs> {obj} in {container} ; {container} at {location} ; "
                        f"{container} closed <action> open {container} "
                        f"<next_obs> {obj} visible at {location}"
                    ),
                ),
                MixedDocument(
                    id=f"env_pair_natural_{suffix}",
                    modality="environment_natural",
                    content=(
                        f"The {obj} is in the {container} at the {location}. "
                        f"Opening the {container} makes the {obj} visible at the {location}."
                    ),
                ),
            ]
        )
    return documents


def render_document(document: MixedDocument) -> str:
    _validate_renderable_document(document)
    return f"<doc type={document.modality} id={document.id}>\n{document.content}\n</doc>\n"


def render_corpus(documents: list[MixedDocument]) -> str:
    return "\n".join(render_document(document) for document in documents)


def load_mixed_documents_jsonl(path: str | Path) -> list[MixedDocument]:
    documents: list[MixedDocument] = []
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
        missing_fields = {"id", "modality", "content"} - record.keys()
        if missing_fields:
            fields = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Invalid JSONL record at line {line_number}: missing required fields: {fields}"
            )
        for field in ("id", "modality", "content"):
            if not isinstance(record[field], str):
                raise ValueError(
                    f"Invalid JSONL record at line {line_number}: field {field} must be a string"
                )
        documents.append(
            MixedDocument(
                id=record["id"],
                modality=record["modality"],
                content=record["content"],
            )
        )
    return documents


def write_mixed_documents_jsonl(path: str | Path, documents: list[MixedDocument]) -> None:
    lines = [
        json.dumps(
            {
                "id": document.id,
                "modality": document.modality,
                "content": document.content,
            },
            ensure_ascii=False,
        )
        for document in documents
    ]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _validate_renderable_document(document: MixedDocument) -> None:
    for field_name, value in (("id", document.id), ("modality", document.modality)):
        if not value or any(character.isspace() for character in value) or ">" in value:
            raise ValueError(f"document {field_name} is not renderable as a tag attribute")
    if "</doc>" in document.content:
        raise ValueError("document content must not contain </doc>")


def default_mixed_documents() -> list[MixedDocument]:
    documents = [
        MixedDocument(
            id="ja_explain_001",
            modality="text",
            content="箱を開けると、中にある物体を観測できる。観測は行動の結果で変化する。",
        ),
        MixedDocument(
            id="en_explain_001",
            modality="text",
            content="A world model uses observations and actions to predict what will be observed next.",
        ),
        MixedDocument(
            id="env_symbolic_001",
            modality="environment_symbolic",
            content="<obs> key in box ; box closed <action> open box <next_obs> key visible",
        ),
        MixedDocument(
            id="env_natural_001",
            modality="environment_natural",
            content="鍵は箱の中にある。箱を開けると、鍵が見える。",
        ),
        MixedDocument(
            id="env_symbolic_002",
            modality="environment_symbolic",
            content="<obs> agent at desk ; cup on desk <action> move cup shelf <next_obs> cup on shelf",
        ),
        MixedDocument(
            id="env_natural_002",
            modality="environment_natural",
            content="机の上にカップがある。カップを棚へ移動すると、次の観測ではカップは棚にある。",
        ),
        MixedDocument(
            id="code_001",
            modality="code",
            content="def move(obj, target):\n    return f'{obj} is at {target}'",
        ),
        MixedDocument(
            id="log_001",
            modality="log",
            content="[tool] action=open_box status=ok observation=key_visible",
        ),
    ]
    return documents + generate_environment_document_pairs()
