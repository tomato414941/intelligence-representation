from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BestMetricSnapshot:
    step: int
    value: float


class BestMetricTracker:
    def __init__(self, *, mode: Literal["min", "max"]) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        self.mode = mode
        self.best: BestMetricSnapshot | None = None

    def update(self, *, step: int, value: float) -> bool:
        if self.best is None or self._is_better(value, self.best.value):
            self.best = BestMetricSnapshot(step=step, value=value)
            return True
        return False

    def _is_better(self, value: float, best_value: float) -> bool:
        if self.mode == "min":
            return value < best_value
        return value > best_value
