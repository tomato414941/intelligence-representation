from __future__ import annotations

from pathlib import Path

import numpy as np


def read_portable_image(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    tokens, offset = _read_ppm_header(data)
    magic, width_text, height_text, max_value_text = tokens
    width = int(width_text)
    height = int(height_text)
    max_value = int(max_value_text)
    if width <= 0 or height <= 0:
        raise ValueError("image width and height must be positive")
    if max_value <= 0 or max_value > 255:
        raise ValueError("portable image max value must be between 1 and 255")

    if magic in ("P6", "P5"):
        channel_count = 3 if magic == "P6" else 1
        expected_size = width * height * channel_count
        payload = data[offset : offset + expected_size]
        if len(payload) != expected_size:
            raise ValueError("portable image payload size does not match header")
        array = np.frombuffer(payload, dtype=np.uint8)
    elif magic in ("P3", "P2"):
        values = data[offset:].decode("ascii").split()
        channel_count = 3 if magic == "P3" else 1
        expected_size = width * height * channel_count
        if len(values) != expected_size:
            raise ValueError("portable image payload size does not match header")
        array = np.array([int(value) for value in values], dtype=np.uint8)
    else:
        raise ValueError("unsupported portable image format")

    if max_value != 255:
        array = np.rint(array.astype(np.float64) * 255.0 / max_value).astype(np.uint8)

    if magic in ("P6", "P3"):
        return array.reshape(height, width, 3)
    return array.reshape(height, width)


def _read_ppm_header(data: bytes) -> tuple[tuple[str, str, str, str], int]:
    tokens: list[str] = []
    index = 0
    while len(tokens) < 4:
        index = _skip_ppm_whitespace_and_comments(data, index)
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        if start == index:
            raise ValueError("portable image header is incomplete")
        tokens.append(data[start:index].decode("ascii"))
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise ValueError("portable image header is incomplete")
    index += 1
    return (tokens[0], tokens[1], tokens[2], tokens[3]), index


def _skip_ppm_whitespace_and_comments(data: bytes, index: int) -> int:
    while index < len(data):
        if data[index] in b" \t\r\n":
            index += 1
            continue
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        break
    return index


def normalized_image_from_path(path: str | Path) -> np.ndarray:
    """Read a portable image as float32 values in 0..1."""
    image = read_portable_image(path)
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3):
        return image.astype(np.float32) / 255.0
    raise ValueError("image payload must be grayscale or RGB")


def channel_count_from_image_shape(image_shape: tuple[int, ...]) -> int:
    if len(image_shape) == 2:
        return 1
    if len(image_shape) == 3:
        return image_shape[2]
    raise ValueError("image shape must be [height, width] or [height, width, channels]")
