from __future__ import annotations

import torch

from intrep.grid_world import GRID_ACTIONS, GridAction, GridObservation, Position, coerce_action


def grid_action_to_id(action: GridAction | str) -> int:
    action = coerce_action(action)
    try:
        return GRID_ACTIONS.index(action.direction)
    except ValueError as error:
        raise ValueError(f"unknown grid action: {action.direction}") from error


def grid_action_from_id(action_id: int) -> GridAction:
    if not 0 <= action_id < len(GRID_ACTIONS):
        raise ValueError("grid action id out of range")
    return GridAction(direction=GRID_ACTIONS[action_id])


def grid_observation_to_tensor(observation: GridObservation) -> torch.Tensor:
    if not observation.grid:
        raise ValueError("observation grid must not be empty")
    width = len(observation.grid[0])
    if width == 0:
        raise ValueError("observation grid rows must not be empty")
    if any(len(row) != width for row in observation.grid):
        raise ValueError("observation grid rows must have the same width")

    tensor = torch.zeros((3, len(observation.grid), width), dtype=torch.float32)
    for row_index, row in enumerate(observation.grid):
        for col_index, cell in enumerate(row):
            if cell in {"A", "*"}:
                tensor[0, row_index, col_index] = 1.0
            elif cell == "G":
                tensor[1, row_index, col_index] = 1.0
            elif cell == "#":
                tensor[2, row_index, col_index] = 1.0
            elif cell != ".":
                raise ValueError(f"unknown grid observation cell: {cell}")
    return tensor


def grid_position_to_cell_id(position: Position, *, width: int) -> int:
    if width <= 0:
        raise ValueError("width must be positive")
    return position.row * width + position.col
