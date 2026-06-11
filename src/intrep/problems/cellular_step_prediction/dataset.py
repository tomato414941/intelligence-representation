from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from intrep.worlds.cellular.world import CellularTransition, CellularWorldState


def cellular_state_to_tensor(state: CellularWorldState) -> torch.Tensor:
    return torch.tensor(state.grid, dtype=torch.float32).unsqueeze(0)


def cellular_state_to_cell_ids(state: CellularWorldState) -> torch.Tensor:
    return torch.tensor(state.grid, dtype=torch.long).reshape(-1)


class CellularStepPredictionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Returns (current observation, next observation per-cell targets)."""

    def __init__(self, transitions: Sequence[CellularTransition]) -> None:
        if not transitions:
            raise ValueError("transitions must not be empty")
        self.height = transitions[0].state.height
        self.width = transitions[0].state.width
        if any(
            transition.state.height != self.height or transition.state.width != self.width
            for transition in transitions
        ):
            raise ValueError("all transitions must use the same grid size")
        self.transitions = tuple(transitions)

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.transitions[index]
        return (
            cellular_state_to_tensor(transition.state),
            cellular_state_to_cell_ids(transition.next_state),
        )
