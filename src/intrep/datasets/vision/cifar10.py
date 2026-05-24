from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np


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


def read_cifar10_images_and_labels(
    batch_paths: Sequence[str | Path],
    *,
    limit: int | None = None,
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


def write_ppm(path: str | Path, pixels: np.ndarray) -> None:
    image = np.asarray(pixels)
    if image.dtype != np.uint8 or image.shape != (32, 32, 3):
        raise ValueError("PPM output requires a uint8 CIFAR-10 RGB image")
    header = b"P6\n32 32\n255\n"
    Path(path).write_bytes(header + image.tobytes())
