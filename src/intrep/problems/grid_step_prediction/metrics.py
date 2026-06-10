from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from intrep.domains.grid.encoding import grid_observation_to_cell_class_ids, grid_position_to_cell_id
from intrep.domains.grid.world import GridExperienceTransition


@dataclass(frozen=True)
class NextObservationMetrics:
    per_cell_accuracy: float
    changed_cell_accuracy: float | None
    whole_grid_match: float
    next_agent_cell_accuracy: float


def next_observation_metrics(
    predicted_class_ids: torch.Tensor,
    agent_scores: torch.Tensor,
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
) -> NextObservationMetrics:
    """Score predicted next observations against the true transitions.

    `predicted_class_ids` has shape [batch, cells]; `agent_scores` has shape
    [batch, cells] and ranks cells by how strongly the prediction places the
    agent there. The next-agent-cell metric is derived from `agent_scores`
    rather than trained as a separate target.
    """
    if not examples:
        raise ValueError("examples must not be empty")
    if predicted_class_ids.shape != agent_scores.shape:
        raise ValueError("predicted_class_ids and agent_scores must have the same shape")
    if predicted_class_ids.size(0) != len(examples):
        raise ValueError("predictions and examples must have the same length")

    target_class_ids = torch.stack(
        [grid_observation_to_cell_class_ids(example.next_observation) for example in examples]
    )
    current_class_ids = torch.stack(
        [grid_observation_to_cell_class_ids(example.observation) for example in examples]
    )
    correct = predicted_class_ids == target_class_ids
    changed = target_class_ids != current_class_ids

    per_cell_accuracy = correct.float().mean().item()
    changed_count = int(changed.sum().item())
    changed_cell_accuracy = (
        correct[changed].float().mean().item() if changed_count > 0 else None
    )
    whole_grid_match = correct.all(dim=1).float().mean().item()

    predicted_agent_cells = agent_scores.argmax(dim=1)
    true_agent_cells = torch.tensor(
        [
            grid_position_to_cell_id(example.next_observation.agent, width=width)
            for example in examples
        ],
        dtype=torch.long,
    )
    next_agent_cell_accuracy = (predicted_agent_cells == true_agent_cells).float().mean().item()

    return NextObservationMetrics(
        per_cell_accuracy=per_cell_accuracy,
        changed_cell_accuracy=changed_cell_accuracy,
        whole_grid_match=whole_grid_match,
        next_agent_cell_accuracy=next_agent_cell_accuracy,
    )
