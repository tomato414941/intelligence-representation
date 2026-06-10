from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from intrep.domains.grid.encoding import grid_observation_to_cell_class_ids, grid_position_to_cell_id
from intrep.domains.grid.world import ACTION_DELTAS, GridExperienceTransition

_AGENT_CLASS_ID = 1
_EMPTY_CLASS_ID = 0


@dataclass(frozen=True)
class BaselinePrediction:
    """Predicted next observations in the same shape model outputs use.

    `class_ids` has shape [batch, cells]; `agent_scores` has shape
    [batch, cells] and feeds the derived next-agent-cell metric.
    """

    class_ids: torch.Tensor
    agent_scores: torch.Tensor


def copy_baseline(examples: Sequence[GridExperienceTransition], *, width: int) -> BaselinePrediction:
    """Predict no change: the next observation equals the current one."""
    class_ids = torch.stack([grid_observation_to_cell_class_ids(example.observation) for example in examples])
    agent_scores = torch.zeros_like(class_ids, dtype=torch.float32)
    for index, example in enumerate(examples):
        agent_scores[index, grid_position_to_cell_id(example.observation.agent, width=width)] = 1.0
    return BaselinePrediction(class_ids=class_ids, agent_scores=agent_scores)


def naive_action_apply_baseline(
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
) -> BaselinePrediction:
    """Move the agent by the action delta whenever the target is in bounds.

    Walls are ignored: the agent overwrites whatever the target cell held, and
    the vacated cell becomes empty. This is deliberately rule-blind.
    """
    class_ids = torch.stack([grid_observation_to_cell_class_ids(example.observation) for example in examples])
    height = class_ids.size(1) // width
    agent_scores = torch.zeros_like(class_ids, dtype=torch.float32)
    for index, example in enumerate(examples):
        agent = example.observation.agent
        delta = ACTION_DELTAS[example.action.direction]
        target_row = agent.row + delta.row
        target_col = agent.col + delta.col
        current_cell = grid_position_to_cell_id(agent, width=width)
        if 0 <= target_row < height and 0 <= target_col < width:
            target_cell = target_row * width + target_col
        else:
            target_cell = current_cell
        if target_cell != current_cell:
            class_ids[index, current_cell] = _EMPTY_CLASS_ID
            class_ids[index, target_cell] = _AGENT_CLASS_ID
        agent_scores[index, target_cell] = 1.0
    return BaselinePrediction(class_ids=class_ids, agent_scores=agent_scores)


@dataclass(frozen=True)
class PerCellMajorityTable:
    majority_class_ids: torch.Tensor
    agent_frequencies: torch.Tensor


def fit_per_cell_majority(
    train_examples: Sequence[GridExperienceTransition],
) -> PerCellMajorityTable:
    """Fit, per cell, the majority next-observation class over training data.

    Ignores the current observation and action entirely. Agent scores are the
    empirical per-cell frequency of the agent class in next observations.
    """
    if not train_examples:
        raise ValueError("train_examples must not be empty")
    target_class_ids = torch.stack(
        [grid_observation_to_cell_class_ids(example.next_observation) for example in train_examples]
    )
    cells = target_class_ids.size(1)
    counts = torch.zeros((cells, 4), dtype=torch.long)
    for cell in range(cells):
        counts[cell] = torch.bincount(target_class_ids[:, cell], minlength=4)
    return PerCellMajorityTable(
        majority_class_ids=counts.argmax(dim=1),
        agent_frequencies=counts[:, _AGENT_CLASS_ID].float() / len(train_examples),
    )


def per_cell_majority_baseline(
    table: PerCellMajorityTable,
    examples: Sequence[GridExperienceTransition],
) -> BaselinePrediction:
    count = len(examples)
    return BaselinePrediction(
        class_ids=table.majority_class_ids.unsqueeze(0).expand(count, -1).clone(),
        agent_scores=table.agent_frequencies.unsqueeze(0).expand(count, -1).clone(),
    )
