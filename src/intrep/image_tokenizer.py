from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from intrep.signals import PayloadRef


@dataclass(frozen=True)
class ImagePatchTokenizer:
    patch_size: int = 1
    channel_bins: int = 4

    @property
    def pad_id(self) -> int:
        return self.channel_bins**3

    @property
    def vocab_size(self) -> int:
        return self.pad_id + 1

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.channel_bins <= 1:
            raise ValueError("channel_bins must be greater than 1")

    def encode_pixels(self, pixels: np.ndarray) -> list[int]:
        rgb = _as_rgb_uint8(pixels)
        height, width, _channels = rgb.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")

        token_ids: list[int] = []
        for row in range(0, height, self.patch_size):
            for col in range(0, width, self.patch_size):
                patch = rgb[row : row + self.patch_size, col : col + self.patch_size]
                mean_rgb = patch.reshape(-1, 3).mean(axis=0)
                token_ids.append(self._quantize_rgb(mean_rgb))
        return token_ids

    def encode_ref(self, ref: PayloadRef) -> list[int]:
        path = _local_path_from_ref(ref)
        pixels = read_portable_image(path)
        return self.encode_pixels(pixels)

    def _quantize_rgb(self, rgb: np.ndarray) -> int:
        bins = np.floor(rgb.astype(np.float64) * self.channel_bins / 256.0).astype(int)
        bins = np.clip(bins, 0, self.channel_bins - 1)
        red, green, blue = (int(value) for value in bins)
        return (red * self.channel_bins + green) * self.channel_bins + blue


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


def _as_rgb_uint8(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise ValueError("image pixels must use dtype uint8")
    if array.ndim == 2:
        return np.repeat(array[:, :, np.newaxis], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    raise ValueError("image pixels must be HxW grayscale or HxWx3 RGB")


def _local_path_from_ref(ref: PayloadRef) -> Path:
    parsed = urlparse(ref.uri)
    if parsed.scheme != "file":
        raise ValueError("image tokenizer currently supports file:// payload refs only")
    if not ref.media_type.startswith("image/"):
        raise ValueError("image tokenizer requires an image/* media_type")
    return Path(parsed.path)


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
    index = _skip_ppm_whitespace_and_comments(data, index)
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
