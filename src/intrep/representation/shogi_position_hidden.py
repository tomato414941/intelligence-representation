from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShogiPositionHiddenLayout:
    state_element_index: int
    square_element_offset: int
    square_element_count: int

