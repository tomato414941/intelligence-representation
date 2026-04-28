from __future__ import annotations

from typing import Literal

from intrep.signals import PayloadRef, Signal, render_payload_text

ImageTokenFormat = Literal["flat", "grid"]


def render_image_token_payload(
    event: Signal,
    *,
    patch_size: int = 1,
    channel_bins: int = 4,
    token_format: ImageTokenFormat = "flat",
) -> str:
    if event.channel != "image" or not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)
    token_ids = encode_image_payload_ref(
        event.payload,
        patch_size=patch_size,
        channel_bins=channel_bins,
    )
    if token_format == "flat":
        return " ".join(str(token_id) for token_id in token_ids)
    if token_format == "grid":
        width, height = image_patch_grid_shape(event.payload, patch_size=patch_size)
        rows = []
        for row in range(height):
            row_tokens = [
                f"r{row}c{col}:{token_ids[row * width + col]}"
                for col in range(width)
            ]
            rows.append(" ".join(row_tokens))
        return "\n".join(rows)
    raise ValueError(f"unsupported image token format: {token_format}")


def render_image_token_document(
    event: Signal,
    *,
    patch_size: int = 1,
    channel_bins: int = 4,
    token_format: ImageTokenFormat = "flat",
) -> str:
    if event.channel != "image" or not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)
    token_text = render_image_token_payload(
        event,
        patch_size=patch_size,
        channel_bins=channel_bins,
        token_format=token_format,
    )
    return (
        f'<IMAGE_TOKENS patch_size="{patch_size}" channel_bins="{channel_bins}" format="{token_format}">\n'
        f"{token_text}\n"
        "</IMAGE_TOKENS>\n"
    )


def encode_image_payload_ref(
    payload_ref: PayloadRef,
    *,
    patch_size: int = 1,
    channel_bins: int = 4,
) -> list[int]:
    from intrep.image_tokenizer import ImagePatchTokenizer

    return ImagePatchTokenizer(patch_size=patch_size, channel_bins=channel_bins).encode_ref(
        payload_ref
    )


def image_patch_grid_shape(payload_ref: PayloadRef, *, patch_size: int) -> tuple[int, int]:
    from intrep.image_tokenizer import read_portable_image

    path = _file_payload_ref_path(payload_ref)
    pixels = read_portable_image(path)
    height, width = pixels.shape[:2]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image dimensions must be divisible by patch_size")
    return width // patch_size, height // patch_size


def _file_payload_ref_path(payload_ref: PayloadRef) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(payload_ref.uri)
    if parsed.scheme != "file":
        raise ValueError("image tokenizer currently supports file:// payload refs only")
    return parsed.path
