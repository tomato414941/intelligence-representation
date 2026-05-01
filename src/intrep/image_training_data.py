from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from intrep.image_io import read_portable_image

T = TypeVar("T")


def image_tensor_from_path(path: Path) -> torch.Tensor:
    image = read_portable_image(path)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3):
        return torch.tensor(image.astype(np.float32) / 255.0, dtype=torch.float32)
    raise ValueError("image payload must be grayscale or RGB")


def channel_count_from_image_shape(image_shape: tuple[int, ...]) -> int:
    if len(image_shape) == 2:
        return 1
    if len(image_shape) == 3:
        return image_shape[2]
    raise ValueError("image shape must be [height, width] or [height, width, channels]")


def seeded_data_loader(
    dataset: Dataset[T],
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[T]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
