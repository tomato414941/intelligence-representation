from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from intrep.problems.cellular_step_prediction.dataset import cellular_state_to_cell_ids
from intrep.worlds.cellular.world import CellularTransition


@dataclass(frozen=True)
class CellularStepPredictionScores:
    """Cheat-resistant score pair plus overall per-cell accuracy.

    A copy strategy scores zero on changed cells; an all-flip strategy scores
    zero on unchanged cells. Both must be high at once to indicate that the
    update rule was actually acquired.
    """

    changed_cell_accuracy: float | None
    unchanged_cell_accuracy: float | None
    per_cell_accuracy: float


def cellular_step_prediction_scores(
    predicted_cell_ids: torch.Tensor,
    transitions: Sequence[CellularTransition],
) -> CellularStepPredictionScores:
    if not transitions:
        raise ValueError("transitions must not be empty")
    if predicted_cell_ids.size(0) != len(transitions):
        raise ValueError("predictions and transitions must have the same length")

    targets = torch.stack([cellular_state_to_cell_ids(transition.next_state) for transition in transitions])
    currents = torch.stack([cellular_state_to_cell_ids(transition.state) for transition in transitions])
    correct = predicted_cell_ids == targets
    changed = targets != currents

    changed_count = int(changed.sum().item())
    unchanged_count = int((~changed).sum().item())
    return CellularStepPredictionScores(
        changed_cell_accuracy=(
            correct[changed].float().mean().item() if changed_count > 0 else None
        ),
        unchanged_cell_accuracy=(
            correct[~changed].float().mean().item() if unchanged_count > 0 else None
        ),
        per_cell_accuracy=correct.float().mean().item(),
    )
