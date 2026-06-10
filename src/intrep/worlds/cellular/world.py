from __future__ import annotations

import random
from dataclasses import dataclass

_NEIGHBOR_OFFSETS = tuple(
    (row, col) for row in (-1, 0, 1) for col in (-1, 0, 1) if (row, col) != (0, 0)
)


@dataclass(frozen=True)
class CellularRule:
    """Outer-totalistic rule: a cell's next state depends only on its own
    state and the count of alive cells among its 8 neighbors.

    Life is birth={3}, survival={2, 3}.
    """

    birth: frozenset[int]
    survival: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "birth", frozenset(self.birth))
        object.__setattr__(self, "survival", frozenset(self.survival))
        for name, counts in (("birth", self.birth), ("survival", self.survival)):
            if any(count < 0 or count > 8 for count in counts):
                raise ValueError(f"{name} counts must be within 0..8")


LIFE_RULE = CellularRule(birth=frozenset({3}), survival=frozenset({2, 3}))


@dataclass(frozen=True)
class CellularWorldState:
    width: int
    height: int
    grid: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        if len(self.grid) != self.height or any(len(row) != self.width for row in self.grid):
            raise ValueError("grid shape must match width and height")
        if any(cell not in (0, 1) for row in self.grid for cell in row):
            raise ValueError("grid cells must be 0 or 1")


def step_cellular_state(state: CellularWorldState, rule: CellularRule) -> CellularWorldState:
    """Apply one synchronous update. Cells outside the grid count as dead."""
    next_rows = []
    for row in range(state.height):
        next_row = []
        for col in range(state.width):
            alive_neighbors = 0
            for row_offset, col_offset in _NEIGHBOR_OFFSETS:
                neighbor_row = row + row_offset
                neighbor_col = col + col_offset
                if 0 <= neighbor_row < state.height and 0 <= neighbor_col < state.width:
                    alive_neighbors += state.grid[neighbor_row][neighbor_col]
            if state.grid[row][col] == 1:
                next_row.append(1 if alive_neighbors in rule.survival else 0)
            else:
                next_row.append(1 if alive_neighbors in rule.birth else 0)
        next_rows.append(tuple(next_row))
    return CellularWorldState(width=state.width, height=state.height, grid=tuple(next_rows))


@dataclass(frozen=True)
class CellularTransition:
    id: str
    rule: CellularRule
    state: CellularWorldState
    next_state: CellularWorldState


def generate_random_cellular_rule(seed: int) -> CellularRule:
    """Sample a rule from the outer-totalistic family.

    Birth on zero neighbors is excluded so the all-dead state stays quiescent.
    """
    rng = random.Random(seed)
    birth = frozenset(count for count in range(1, 9) if rng.random() < 0.5)
    survival = frozenset(count for count in range(0, 9) if rng.random() < 0.5)
    return CellularRule(birth=birth, survival=survival)


def generate_random_cellular_state(
    *,
    width: int,
    height: int,
    alive_probability: float = 0.5,
    seed: int,
) -> CellularWorldState:
    if not 0.0 <= alive_probability <= 1.0:
        raise ValueError("alive_probability must be within 0..1")
    rng = random.Random(seed)
    grid = tuple(
        tuple(1 if rng.random() < alive_probability else 0 for _ in range(width))
        for _ in range(height)
    )
    return CellularWorldState(width=width, height=height, grid=grid)


def generate_cellular_transitions(
    rule: CellularRule,
    *,
    width: int,
    height: int,
    count: int,
    alive_probability: float = 0.5,
    seed: int,
) -> list[CellularTransition]:
    """Generate one-step transitions from independent random initial states."""
    if count <= 0:
        raise ValueError("count must be positive")
    transitions = []
    for index in range(count):
        state = generate_random_cellular_state(
            width=width,
            height=height,
            alive_probability=alive_probability,
            seed=seed + index,
        )
        transitions.append(
            CellularTransition(
                id=f"cellular_transition_{index + 1}",
                rule=rule,
                state=state,
                next_state=step_cellular_state(state, rule),
            )
        )
    return transitions
