from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


FASHION_MNIST_LABELS = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

MNIST_LABELS = tuple(str(index) for index in range(10))

CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class ImageClassificationExample:
    image_path: Path
    label_names: tuple[str, ...]
    label_index: int

    def __post_init__(self) -> None:
        if not self.label_names:
            raise ValueError("label_names must not be empty")
        if not 0 <= self.label_index < len(self.label_names):
            raise ValueError("label_index out of range")

    @property
    def label_text(self) -> str:
        return self.label_names[self.label_index]


def load_image_classification_examples_jsonl(path: str | Path) -> list[ImageClassificationExample]:
    examples: list[ImageClassificationExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-classification JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_classification_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-classification JSONL must contain at least one example")
    return examples


def image_classification_example_from_record(
    record: object,
    *,
    line_number: int,
) -> ImageClassificationExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: expected object")
    required = {"image_path", "label_names", "label_index"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    label_names = record["label_names"]
    label_index = record["label_index"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(
            f"Invalid image-classification JSONL at line {line_number}: image_path must be a string"
        )
    if not isinstance(label_names, list) or not all(isinstance(label_name, str) for label_name in label_names):
        raise ValueError(
            f"Invalid image-classification JSONL at line {line_number}: label_names must be a list of strings"
        )
    if not isinstance(label_index, int):
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: label_index must be an integer")
    try:
        return ImageClassificationExample(
            image_path=Path(image_path),
            label_names=tuple(label_names),
            label_index=label_index,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: {error}") from error


def image_classification_example_to_record(example: ImageClassificationExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "label_names": list(example.label_names),
        "label_index": example.label_index,
    }


def class_count_from_examples(examples: list[ImageClassificationExample]) -> int:
    if not examples:
        raise ValueError("examples must not be empty")
    label_names = examples[0].label_names
    for example in examples:
        if example.label_names != label_names:
            raise ValueError("all examples must use the same label_names")
    return len(label_names)
