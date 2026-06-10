from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


MOVING_DOT_ACTIONS: tuple[str, ...] = ("stay", "up", "down", "left", "right")


@dataclass(frozen=True)
class MovingDotTransition:
    frame: NDArray[np.float32]
    action: int
    next_frame: NDArray[np.float32]


class MovingDotTransitionDataset:
    def __init__(
        self,
        *,
        sample_count: int,
        image_size: int = 16,
        seed: int = 0,
    ) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        if image_size < 2:
            raise ValueError("image_size must be at least 2")

        generator = np.random.default_rng(seed)

        self._image_size = image_size
        self._positions = generator.integers(0, image_size, size=(sample_count, 2), dtype=np.int64)
        self._actions = generator.integers(0, len(MOVING_DOT_ACTIONS), size=sample_count, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._actions.size)

    def __getitem__(self, index: int) -> MovingDotTransition:
        position = self._positions[index]
        action = int(self._actions[index])
        next_position = moving_dot_next_position(position, action, image_size=self._image_size)
        return MovingDotTransition(
            frame=moving_dot_frame(position, image_size=self._image_size),
            action=action,
            next_frame=moving_dot_frame(next_position, image_size=self._image_size),
        )


def moving_dot_next_position(position: NDArray[np.integer], action: int, *, image_size: int) -> NDArray[np.int64]:
    if action < 0 or action >= len(MOVING_DOT_ACTIONS):
        raise ValueError("action is out of range")

    row = int(position[0])
    col = int(position[1])
    if action == 1:
        row -= 1
    elif action == 2:
        row += 1
    elif action == 3:
        col -= 1
    elif action == 4:
        col += 1

    row = min(max(row, 0), image_size - 1)
    col = min(max(col, 0), image_size - 1)
    return np.array([row, col], dtype=np.int64)


def moving_dot_frame(position: NDArray[np.integer], *, image_size: int) -> NDArray[np.float32]:
    frame = np.zeros((1, image_size, image_size), dtype=np.float32)
    frame[0, int(position[0]), int(position[1])] = 1.0
    return frame
