from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from intrep.image_classification import (
    CIFAR10_LABELS,
    ImageTextChoiceExample,
    ImageClassificationExample,
    image_classification_example_to_record,
    image_text_choice_example_to_record,
)


@dataclass(frozen=True)
class CIFAR10Selection:
    examples: list[ImageTextChoiceExample]
    image_count: int
    output_dir: Path


@dataclass(frozen=True)
class CIFAR10ClassificationSelection:
    examples: list[ImageClassificationExample]
    image_count: int
    output_dir: Path


def write_cifar10_image_classification_jsonl(
    *,
    batch_paths: Sequence[str | Path],
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> CIFAR10ClassificationSelection:
    images, labels = _read_cifar10_images_and_labels(batch_paths, limit=limit)
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

    lines = [
        json.dumps(image_classification_example_to_record(example), ensure_ascii=False)
        for example in examples
    ]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return CIFAR10ClassificationSelection(examples=examples, image_count=len(examples), output_dir=output_dir)


def write_cifar10_image_choice_jsonl(
    *,
    batch_paths: Sequence[str | Path],
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> CIFAR10Selection:
    images, labels = _read_cifar10_images_and_labels(batch_paths, limit=limit)
    output_dir = Path(image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples: list[ImageTextChoiceExample] = []
    for index, (image, label_id) in enumerate(zip(images, labels, strict=True)):
        if not 0 <= label_id < len(CIFAR10_LABELS):
            raise ValueError("label id is out of range for CIFAR-10 labels")
        image_path = output_dir / f"{split}_{index:06d}.ppm"
        write_ppm(image_path, image)
        examples.append(
            ImageTextChoiceExample(
                image_path=image_path.resolve(),
                choices=CIFAR10_LABELS,
                answer_index=label_id,
            )
        )

    lines = [
        json.dumps(image_text_choice_example_to_record(example), ensure_ascii=False)
        for example in examples
    ]
    Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return CIFAR10Selection(examples=examples, image_count=len(examples), output_dir=output_dir)


def _read_cifar10_images_and_labels(
    batch_paths: Sequence[str | Path],
    *,
    limit: int | None,
) -> tuple[list[np.ndarray], list[int]]:
    if not batch_paths:
        raise ValueError("batch_paths must not be empty")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    images: list[np.ndarray] = []
    labels: list[int] = []
    for batch_path in batch_paths:
        batch_images, batch_labels = read_cifar10_batch(batch_path)
        images.extend(batch_images)
        labels.extend(batch_labels)

    count = len(images) if limit is None else min(limit, len(images))
    return images[:count], labels[:count]


def read_cifar10_batch(path: str | Path) -> tuple[list[np.ndarray], list[int]]:
    with Path(path).open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = pickle.load(handle, encoding="bytes")
    if not isinstance(batch, dict):
        raise ValueError("CIFAR-10 batch must be a dictionary")
    data = batch.get(b"data")
    labels = batch.get(b"labels")
    if data is None or labels is None:
        raise ValueError("CIFAR-10 batch must contain data and labels")
    data_array = np.asarray(data, dtype=np.uint8)
    label_list = [int(label) for label in labels]
    if data_array.ndim != 2 or data_array.shape[1] != 3072:
        raise ValueError("CIFAR-10 data must have shape [count, 3072]")
    if data_array.shape[0] != len(label_list):
        raise ValueError("CIFAR-10 image and label counts must match")
    images = data_array.reshape(data_array.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    return [image.copy() for image in images], label_list


def write_ppm(path: str | Path, pixels: np.ndarray) -> None:
    image = np.asarray(pixels)
    if image.dtype != np.uint8 or image.shape != (32, 32, 3):
        raise ValueError("PPM output requires a uint8 CIFAR-10 RGB image")
    header = b"P6\n32 32\n255\n"
    Path(path).write_bytes(header + image.tobytes())


def main(argv: Sequence[str] | None = None) -> None:
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
    print("intrep cifar10 image corpus")
    print(f"images={selection.image_count}")
    print(f"examples={len(selection.examples)}")
    print(f"output_path={args.output_path}")
    print(f"image_output_dir={selection.output_dir}")


if __name__ == "__main__":
    main()
