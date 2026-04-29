from __future__ import annotations

from intrep.signals import Signal, render_payload_text
from intrep.signal_stream import render_signal_stream


def render_signal_corpus(events: list[Signal]) -> str:
    return render_signal_stream(events)


def render_signals_for_training(
    events: list[Signal],
    *,
    render_format: str = "signal-tags",
) -> str:
    if render_format == "plain":
        return "\n".join(render_payload_text(event) for event in events) + "\n"
    if render_format not in ("signal-tags", "typed-tags"):
        raise ValueError("render_format must be plain, signal-tags, or typed-tags")
    return render_signal_stream(events)
