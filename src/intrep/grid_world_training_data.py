from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from intrep.grid_world import GridExperienceTransition
from intrep.grid_world_encoding import grid_action_to_id, grid_observation_to_tensor, grid_position_to_cell_id


class GridStepPredictionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: Sequence[GridExperienceTransition]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        first_tensor = grid_observation_to_tensor(examples[0].observation)
        self.grid_shape = tuple(int(value) for value in first_tensor.shape)
        self.height = self.grid_shape[1]
        self.width = self.grid_shape[2]
        self.examples = tuple(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        observation = grid_observation_to_tensor(example.observation)
        if tuple(observation.shape) != self.grid_shape:
            raise ValueError("all grid observations must have the same shape")
        action_id = torch.tensor(grid_action_to_id(example.action), dtype=torch.long)
        next_cell_id = torch.tensor(
            grid_position_to_cell_id(example.next_observation.agent, width=self.width),
            dtype=torch.long,
        )
        reward_id = torch.tensor(grid_reward_to_id(example.reward), dtype=torch.long)
        terminated_id = torch.tensor(int(example.terminated), dtype=torch.long)
        return observation, action_id, next_cell_id, reward_id, terminated_id


def grid_reward_to_id(reward: float) -> int:
    if reward == -0.1:
        return 0
    if reward == -0.01:
        return 1
    if reward == 1.0:
        return 2
    raise ValueError(f"unsupported grid reward: {reward}")
