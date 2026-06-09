from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset

from intrep.domains.grid.encoding import grid_action_to_id, grid_observation_to_tensor, grid_position_to_cell_id
from intrep.domains.grid.world import GridExperienceTransition, Position


class GridStepPredictionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Returns deterministic transition-prediction samples from grid experience.

    The next-cell, reward, and terminated targets are cheap to derive from each
    transition, so this dataset materializes them as runtime samples instead of
    storing separate Training Example records.
    """

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


def split_grid_transitions_by_agent_cell(
    examples: Sequence[GridExperienceTransition],
    *,
    held_out_cells: Sequence[Position],
) -> tuple[list[GridExperienceTransition], list[GridExperienceTransition]]:
    held_out = set(held_out_cells)
    train_examples = []
    eval_examples = []
    for example in examples:
        if example.observation.agent in held_out:
            eval_examples.append(example)
        else:
            train_examples.append(example)
    if not train_examples:
        raise ValueError("train split must not be empty")
    return train_examples, eval_examples


def grid_reward_to_id(reward: float) -> int:
    if reward == -0.1:
        return 0
    if reward == -0.01:
        return 1
    if reward == 1.0:
        return 2
    raise ValueError(f"unsupported grid reward: {reward}")
