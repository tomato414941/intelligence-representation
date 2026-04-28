from __future__ import annotations

from intrep.signals import PayloadRef, Signal, render_payload_text


def render_image_token_payload(
    event: Signal,
    *,
    patch_size: int = 1,
    channel_bins: int = 4,
) -> str:
    if event.channel != "image" or not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)
    return " ".join(
        str(token_id)
        for token_id in encode_image_payload_ref(
            event.payload,
            patch_size=patch_size,
            channel_bins=channel_bins,
        )
    )


def render_image_token_document(
    event: Signal,
    *,
    patch_size: int = 1,
    channel_bins: int = 4,
) -> str:
    if event.channel != "image" or not isinstance(event.payload, PayloadRef):
        return render_payload_text(event)
    token_text = render_image_token_payload(
        event,
        patch_size=patch_size,
        channel_bins=channel_bins,
    )
    return (
        f'<IMAGE_TOKENS patch_size="{patch_size}" channel_bins="{channel_bins}">\n'
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
