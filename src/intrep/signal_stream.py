from __future__ import annotations

from collections.abc import Sequence

from intrep.signals import Signal, render_signal as render_signal_tag


def render_signal_stream(events: Sequence[Signal]) -> str:
    return "\n".join(render_signal(event) for event in events)


def render_signal(event: Signal) -> str:
    return render_signal_tag(event)
