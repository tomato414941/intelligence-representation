from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from intrep.datasets.vision.cifar10 import CIFAR10_LABELS, read_cifar10_images_and_labels, write_ppm
from intrep.datasets.vision.idx import FASHION_MNIST_LABELS, MNIST_LABELS, read_idx_images, read_idx_labels, write_pgm
from intrep.problems.image_classification.examples import (
    ImageClassificationExample,
    image_classification_example_to_record,
)


@dataclass(frozen=True)
class ImageClassificationDatasetBuild:
    examples: list[ImageClassificationExample]
    image_count: int
    output_dir: Path


def write_idx_image_classification_jsonl(
    *,
    images_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
    label_names: Sequence[str] = FASHION_MNIST_LABELS,
) -> ImageClassificationDatasetBuild:
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

    examples: list[ImageClassificationExample] = []
    for index in range(count):
        label_id = int(labels[index])
        if not 0 <= label_id < len(label_names):
            raise ValueError("label id is out of range for label_names")
        image_path = output_dir / f"{split}_{index:06d}.pgm"
        write_pgm(image_path, images[index])
        examples.append(
            ImageClassificationExample(
                image_path=image_path.resolve(),
                label_names=tuple(label_names),
                label_index=label_id,
            )
        )

    _write_jsonl(output_path, [image_classification_example_to_record(example) for example in examples])
    return ImageClassificationDatasetBuild(examples=examples, image_count=count, output_dir=output_dir)


def write_cifar10_image_classification_jsonl(
    *,
    batch_paths: Sequence[str | Path],
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> ImageClassificationDatasetBuild:
    images, labels = read_cifar10_images_and_labels(batch_paths, limit=limit)
    output_dir = Path(image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples: list[ImageClassificationExample] = []
    for index, (image, label_id) in enumerate(zip(images, labels, strict=True)):
        if not 0 <= label_id < len(CIFAR10_LABELS):
            raise ValueError("label id is out of range for CIFAR-10 labels")
        image_path = output_dir / f"{split}_{index:06d}.ppm"
        write_ppm(image_path, image)
        examples.append(
            ImageClassificationExample(
                image_path=image_path.resolve(),
                label_names=CIFAR10_LABELS,
                label_index=label_id,
            )
        )

    _write_jsonl(output_path, [image_classification_example_to_record(example) for example in examples])
    return ImageClassificationDatasetBuild(examples=examples, image_count=len(examples), output_dir=output_dir)


def main_idx(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert local IDX image and label files into image-classification JSONL records."
    )
    parser.add_argument("--images-path", required=True, help="Path to images IDX or IDX gzip.")
    parser.add_argument("--labels-path", required=True, help="Path to labels IDX or IDX gzip.")
    parser.add_argument("--output-path", required=True, help="Path for output image-classification JSONL.")
    parser.add_argument("--image-output-dir", required=True, help="Directory for extracted PGM images.")
    parser.add_argument("--split", default="train", help="Split label used in generated image filenames.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of examples to convert.")
    parser.add_argument(
        "--label-set",
        choices=("fashion-mnist", "mnist"),
        default="fashion-mnist",
        help="Label names to attach to the generated classification examples.",
    )
    args = parser.parse_args(argv)

    selection = write_idx_image_classification_jsonl(
        images_path=args.images_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        image_output_dir=args.image_output_dir,
        split=args.split,
        limit=args.limit,
        label_names=_label_names(args.label_set),
    )
    print("intrep idx image classification dataset")
    print(f"label_set={args.label_set}")
    print(f"images={selection.image_count}")
    print(f"examples={len(selection.examples)}")
    print(f"output_path={args.output_path}")
    print(f"image_output_dir={selection.output_dir}")


def main_cifar10(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert local CIFAR-10 python batch files into image-classification JSONL records."
    )
    parser.add_argument(
        "--batch-path",
        action="append",
        required=True,
        help="Path to a CIFAR-10 python batch file. Repeat for multiple batches.",
    )
    parser.add_argument("--output-path", required=True, help="Path for output image-classification JSONL.")
    parser.add_argument("--image-output-dir", required=True, help="Directory for extracted PPM images.")
    parser.add_argument("--split", default="train", help="Split label used in generated image filenames.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of examples to convert.")
    args = parser.parse_args(argv)

    selection = write_cifar10_image_classification_jsonl(
        batch_paths=args.batch_path,
        output_path=args.output_path,
        image_output_dir=args.image_output_dir,
        split=args.split,
        limit=args.limit,
    )
    print("intrep cifar10 image classification dataset")
    print(f"images={selection.image_count}")
    print(f"examples={len(selection.examples)}")
    print(f"output_path={args.output_path}")
    print(f"image_output_dir={selection.output_dir}")


def _label_names(label_set: str) -> tuple[str, ...]:
    if label_set == "fashion-mnist":
        return FASHION_MNIST_LABELS
    if label_set == "mnist":
        return MNIST_LABELS
    raise ValueError(f"unknown label set: {label_set}")


def _write_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
