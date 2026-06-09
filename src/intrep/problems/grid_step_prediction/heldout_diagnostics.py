from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from intrep.domains.grid.encoding import grid_action_to_id, grid_observation_to_tensor
from intrep.domains.grid.world import (
    GridExperienceTransition,
    GridWorldState,
    Position,
    generate_grid_world_transition_table,
)
from intrep.problems.grid_step_prediction.dataset import split_grid_transitions_by_agent_cell
from intrep.problems.grid_step_prediction.training import (
    GridStepPredictionConfig,
    train_grid_step_predictor_with_artifacts,
)
from intrep.representation.assemblies.grid_step_prediction import GridStepPredictionModel


@dataclass(frozen=True)
class TransitionPrediction:
    agent: Position
    action: str
    true_next: Position
    predicted_next: Position

    @property
    def correct(self) -> bool:
        return self.predicted_next == self.true_next


@dataclass(frozen=True)
class HeldOutCellRun:
    held_out_cell: Position
    seed: int
    train_case_count: int
    eval_case_count: int
    train_next_cell_accuracy: float
    eval_next_cell_accuracy: float
    predictions: tuple[TransitionPrediction, ...]


def predict_next_cells(
    model: GridStepPredictionModel,
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
) -> list[TransitionPrediction]:
    if not examples:
        raise ValueError("examples must not be empty")
    observations = torch.stack([grid_observation_to_tensor(example.observation) for example in examples])
    action_ids = torch.tensor([grid_action_to_id(example.action) for example in examples], dtype=torch.long)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        next_cell_logits, _, _ = model(observations.to(device), action_ids.to(device))
    predicted_cell_ids = next_cell_logits.argmax(dim=1).tolist()
    return [
        TransitionPrediction(
            agent=example.observation.agent,
            action=example.action.direction,
            true_next=example.next_observation.agent,
            predicted_next=Position(row=cell_id // width, col=cell_id % width),
        )
        for example, cell_id in zip(examples, predicted_cell_ids)
    ]


def run_held_out_cell_sweep(
    state_template: GridWorldState,
    *,
    seeds: Sequence[int],
    config: GridStepPredictionConfig,
) -> list[HeldOutCellRun]:
    """Train once per (held-out agent cell, seed) and record per-action predictions.

    Every valid agent cell takes a turn as the held-out cell, so a single split
    or initialization cannot dominate the diagnostic.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    examples = generate_grid_world_transition_table(state_template)
    agent_cells = sorted(
        {example.observation.agent for example in examples},
        key=lambda position: (position.row, position.col),
    )
    runs = []
    for held_out_cell in agent_cells:
        train_examples, eval_examples = split_grid_transitions_by_agent_cell(
            examples,
            held_out_cells=[held_out_cell],
        )
        for seed in seeds:
            artifacts = train_grid_step_predictor_with_artifacts(
                train_examples,
                eval_examples=eval_examples,
                config=replace(config, seed=seed),
            )
            predictions = predict_next_cells(
                artifacts.model,
                eval_examples,
                width=state_template.width,
            )
            result = artifacts.result
            assert result.eval_next_cell_accuracy is not None
            runs.append(
                HeldOutCellRun(
                    held_out_cell=held_out_cell,
                    seed=seed,
                    train_case_count=result.train_case_count,
                    eval_case_count=result.eval_case_count,
                    train_next_cell_accuracy=result.next_cell_accuracy,
                    eval_next_cell_accuracy=result.eval_next_cell_accuracy,
                    predictions=tuple(predictions),
                )
            )
    return runs
