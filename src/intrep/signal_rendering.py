from __future__ import annotations

from intrep.image_rendering import render_image_token_document
from intrep.signals import Signal, render_payload_text
from intrep.signal_stream import render_signal_stream


def render_signal_corpus(events: list[Signal]) -> str:
    return render_signal_stream(events)


def render_signals_for_training(
    events: list[Signal],
    *,
    render_format: str = "signal-tags",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
) -> str:
    if render_format == "plain":
        return "\n".join(render_payload_text(event) for event in events) + "\n"
    if render_format == "image-tokens":
        return "".join(
            render_image_token_document(
                event,
                patch_size=image_patch_size,
                channel_bins=image_channel_bins,
                token_format=image_token_format,
            )
            for event in events
        )
    if render_format not in ("signal-tags", "typed-tags"):
        raise ValueError("render_format must be plain, signal-tags, typed-tags, or image-tokens")
    return render_signal_stream(events)
