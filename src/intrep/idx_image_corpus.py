from __future__ import annotations

import argparse
import gzip
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from intrep.image_classification import (
    FASHION_MNIST_LABELS,
    ImageTextChoiceExample,
    ImageClassificationExample,
    MNIST_LABELS,
    image_classification_example_to_record,
    image_text_choice_example_to_record,
)


@dataclass(frozen=True)
class IDXImageTextChoiceSelection:
    examples: list[ImageTextChoiceExample]
    image_count: int
    output_dir: Path


@dataclass(frozen=True)
class IDXImageClassificationSelection:
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
) -> IDXImageClassificationSelection:
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

    lines = [
        json.dumps(image_classification_example_to_record(example), ensure_ascii=False)
        for example in examples
    ]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return IDXImageClassificationSelection(examples=examples, image_count=count, output_dir=output_dir)


def write_idx_image_text_choice_jsonl(
    *,
    images_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
    label_names: Sequence[str] = FASHION_MNIST_LABELS,
) -> IDXImageTextChoiceSelection:
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

    examples: list[ImageTextChoiceExample] = []
    for index in range(count):
        label_id = int(labels[index])
        if not 0 <= label_id < len(label_names):
            raise ValueError("label id is out of range for label_names")
        image_path = output_dir / f"{split}_{index:06d}.pgm"
        write_pgm(image_path, images[index])
        examples.append(
            ImageTextChoiceExample(
                image_path=image_path.resolve(),
                choices=tuple(label_names),
                answer_index=label_id,
            )
        )

    lines = [
        json.dumps(image_text_choice_example_to_record(example), ensure_ascii=False)
        for example in examples
    ]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return IDXImageTextChoiceSelection(examples=examples, image_count=count, output_dir=output_dir)


def read_idx_images(path: str | Path) -> np.ndarray:
    data = _read_maybe_gzip(path)
    if len(data) < 16:
        raise ValueError("IDX image file is too small")
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError("IDX image file has invalid magic number")
    expected_size = 16 + count * rows * cols
    if len(data) != expected_size:
        raise ValueError("IDX image payload size does not match header")
    pixels = np.frombuffer(data[16:], dtype=np.uint8)
    return pixels.reshape(count, rows, cols)


def read_idx_labels(path: str | Path) -> np.ndarray:
    data = _read_maybe_gzip(path)
    if len(data) < 8:
        raise ValueError("IDX label file is too small")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError("IDX label file has invalid magic number")
    expected_size = 8 + count
    if len(data) != expected_size:
        raise ValueError("IDX label payload size does not match header")
    return np.frombuffer(data[8:], dtype=np.uint8)


def write_pgm(path: str | Path, pixels: np.ndarray) -> None:
    image = np.asarray(pixels)
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("PGM output requires a uint8 grayscale image")
    height, width = image.shape
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    Path(path).write_bytes(header + image.tobytes())


def _read_maybe_gzip(path: str | Path) -> bytes:
    source = Path(path)
    if source.suffix == ".gz":
        with gzip.open(source, "rb") as handle:
            return handle.read()
    return source.read_bytes()


def main(argv: Sequence[str] | None = None) -> None:
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
    print("intrep idx image corpus")
    print(f"label_set={args.label_set}")
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


if __name__ == "__main__":
    main()
