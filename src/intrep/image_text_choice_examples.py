from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageTextChoiceExample:
    image_path: Path
    choices: tuple[str, ...]
    answer_index: int

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("choices must not be empty")
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError("answer_index out of range")

    @property
    def answer_text(self) -> str:
        return self.choices[self.answer_index]


def load_image_text_choice_examples_jsonl(path: str | Path) -> list[ImageTextChoiceExample]:
    examples: list[ImageTextChoiceExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_text_choice_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-text-choice JSONL must contain at least one example")
    return examples


def image_text_choice_example_from_record(record: object, *, line_number: int) -> ImageTextChoiceExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: expected object")
    required = {"image_path", "choices", "answer_index"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    choices = record["choices"]
    answer_index = record["answer_index"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: image_path must be a string")
    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: choices must be a list of strings")
    if not isinstance(answer_index, int):
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: answer_index must be an integer")
    try:
        return ImageTextChoiceExample(
            image_path=Path(image_path),
            choices=tuple(choices),
            answer_index=answer_index,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-text-choice JSONL at line {line_number}: {error}") from error


def image_text_choice_example_to_record(example: ImageTextChoiceExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "choices": list(example.choices),
        "answer_index": example.answer_index,
    }
