from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from intrep.datasets.vision.cifar10 import CIFAR10_LABELS, read_cifar10_images_and_labels, write_ppm
from intrep.datasets.vision.idx import FASHION_MNIST_LABELS, read_idx_images, read_idx_labels, write_pgm
from intrep.problems.image_text_answer.examples import ImageTextAnswerExample, image_text_answer_example_to_record


@dataclass(frozen=True)
class ImageTextAnswerDatasetBuild:
    examples: list[ImageTextAnswerExample]
    image_count: int
    output_dir: Path


def write_idx_image_text_answer_jsonl(
    *,
    images_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    image_output_dir: str | Path,
    prompt: str,
    split: str = "train",
    limit: int | None = None,
    label_names: Sequence[str] = FASHION_MNIST_LABELS,
) -> ImageTextAnswerDatasetBuild:
    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)
    if len(images) != len(labels):
        raise ValueError("IDX image and label counts must match")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if not label_names:
        raise ValueError("label_names must not be empty")
    count = len(images) if limit is None else min(limit, len(images))
    output_dir = Path(image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[ImageTextAnswerExample] = []
    for index in range(count):
        label_id = int(labels[index])
        if not 0 <= label_id < len(label_names):
            raise ValueError("label id is out of range for label_names")
        image_path = output_dir / f"{split}_{index:06d}.pgm"
        write_pgm(image_path, images[index])
        examples.append(
            ImageTextAnswerExample(
                image_path=image_path.resolve(),
                prompt=prompt,
                answer_text=label_names[label_id],
            )
        )
    _write_jsonl(output_path, [image_text_answer_example_to_record(example) for example in examples])
    return ImageTextAnswerDatasetBuild(examples=examples, image_count=count, output_dir=output_dir)


def write_cifar10_image_text_answer_jsonl(
    *,
    batch_paths: Sequence[str | Path],
    output_path: str | Path,
    image_output_dir: str | Path,
    prompt: str,
    split: str = "train",
    limit: int | None = None,
) -> ImageTextAnswerDatasetBuild:
    images, labels = read_cifar10_images_and_labels(batch_paths, limit=limit)
    output_dir = Path(image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[ImageTextAnswerExample] = []
    for index, (image, label_id) in enumerate(zip(images, labels, strict=True)):
        if not 0 <= label_id < len(CIFAR10_LABELS):
            raise ValueError("label id is out of range for CIFAR-10 labels")
        image_path = output_dir / f"{split}_{index:06d}.ppm"
        write_ppm(image_path, image)
        examples.append(
            ImageTextAnswerExample(
                image_path=image_path.resolve(),
                prompt=prompt,
                answer_text=CIFAR10_LABELS[label_id],
            )
        )
    _write_jsonl(output_path, [image_text_answer_example_to_record(example) for example in examples])
    return ImageTextAnswerDatasetBuild(examples=examples, image_count=len(examples), output_dir=output_dir)


def _write_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
