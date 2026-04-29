from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """Reserved name for the retired Signal abstraction."""

    def __post_init__(self) -> None:
        raise RuntimeError(
            "Signal is retired; use task-specific raw example types before tokenization"
        )
