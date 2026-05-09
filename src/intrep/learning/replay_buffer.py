from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class ReplayBuffer(Generic[T]):
    """Dynamic training-time buffer for uniform experience replay."""

    def __init__(self, *, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: deque[T] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, item: T) -> None:
        self._items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        self._items.extend(items)

    def sample(self, batch_size: int, *, generator: torch.Generator | None = None) -> list[T]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError("batch_size must not exceed replay buffer length")
        item_list = list(self._items)
        indices = torch.randperm(len(item_list), generator=generator)[:batch_size].tolist()
        return [item_list[index] for index in indices]
