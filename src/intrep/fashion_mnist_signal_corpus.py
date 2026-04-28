from __future__ import annotations

import argparse
import gzip
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from intrep.signal_corpus import write_signals_jsonl_v2
from intrep.signals import PayloadRef, Signal


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


@dataclass(frozen=True)
class FashionMNISTSelection:
    events: list[Signal]
    image_count: int
    output_dir: Path


def write_fashion_mnist_signal_jsonl(
    *,
    images_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    image_output_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
) -> FashionMNISTSelection:
    images = read_idx_images(images_path)
    labels = read_idx_labels(labels_path)
    if len(images) != len(labels):
        raise ValueError("Fashion-MNIST image and label counts must match")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    count = len(images) if limit is None else min(limit, len(images))
    output_dir = Path(image_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events: list[Signal] = []
    for index in range(count):
        label_id = int(labels[index])
        label_name = _label_name(label_id)
        image_path = output_dir / f"{split}_{index:06d}.pgm"
        write_pgm(image_path, images[index])
        image_bytes = image_path.read_bytes()
        events.extend(
            [
                Signal(
                    channel="image",
                    payload=PayloadRef(
                        uri=image_path.resolve().as_uri(),
                        media_type="image/x-portable-graymap",
                        sha256=hashlib.sha256(image_bytes).hexdigest(),
                        size_bytes=len(image_bytes),
                    ),
                ),
                Signal(
                    channel="label",
                    payload=f"{label_id}:{label_name}",
                ),
            ]
        )

    write_signals_jsonl_v2(output_path, events)
    return FashionMNISTSelection(events=events, image_count=count, output_dir=output_dir)


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


def _label_name(label_id: int) -> str:
    if not 0 <= label_id < len(FASHION_MNIST_LABELS):
        raise ValueError(f"Fashion-MNIST label id out of range: {label_id}")
    return FASHION_MNIST_LABELS[label_id]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert local Fashion-MNIST IDX files into Signal JSONL with image payload_ref records."
    )
    parser.add_argument("--images-path", required=True, help="Path to Fashion-MNIST images IDX or IDX gzip.")
    parser.add_argument("--labels-path", required=True, help="Path to Fashion-MNIST labels IDX or IDX gzip.")
    parser.add_argument("--output-path", required=True, help="Path for output Signal JSONL.")
    parser.add_argument("--image-output-dir", required=True, help="Directory for extracted PGM images.")
    parser.add_argument("--split", default="train", help="Split label used in generated image filenames.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of examples to convert.")
    args = parser.parse_args(argv)

    selection = write_fashion_mnist_signal_jsonl(
        images_path=args.images_path,
        labels_path=args.labels_path,
        output_path=args.output_path,
        image_output_dir=args.image_output_dir,
        split=args.split,
        limit=args.limit,
    )
    print("intrep fashion-mnist signal corpus")
    print(f"images={selection.image_count}")
    print(f"events={len(selection.events)}")
    print(f"output_path={args.output_path}")
    print(f"image_output_dir={selection.output_dir}")


if __name__ == "__main__":
    main()
