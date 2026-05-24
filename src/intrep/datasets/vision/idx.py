from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


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
