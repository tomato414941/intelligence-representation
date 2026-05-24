from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageTextAnswerExample:
    image_path: Path
    prompt: str
    answer_text: str

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.answer_text:
            raise ValueError("answer_text must not be empty")


def load_image_text_answer_examples_jsonl(path: str | Path) -> list[ImageTextAnswerExample]:
    examples: list[ImageTextAnswerExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_text_answer_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-text-answer JSONL must contain at least one example")
    return examples


def image_text_answer_example_from_record(record: object, *, line_number: int) -> ImageTextAnswerExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: expected object")
    required = {"image_path", "prompt", "answer_text"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    prompt = record["prompt"]
    answer_text = record["answer_text"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: image_path must be a string")
    if not isinstance(prompt, str):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: prompt must be a string")
    if not isinstance(answer_text, str):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: answer_text must be a string")
    try:
        return ImageTextAnswerExample(
            image_path=Path(image_path),
            prompt=prompt,
            answer_text=answer_text,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: {error}") from error


def image_text_answer_example_to_record(example: ImageTextAnswerExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "prompt": example.prompt,
        "answer_text": example.answer_text,
    }
