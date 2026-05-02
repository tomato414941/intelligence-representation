from __future__ import annotations

from typing import Sequence

from intrep.grid_world import GridExperienceTransition, GridObservation
from intrep.text_examples import LanguageModelingExample


def grid_observation_to_text(observation: GridObservation) -> str:
    return "\n".join(observation.grid)


def grid_experience_transition_to_text(example: GridExperienceTransition) -> str:
    return "\n".join(
        (
            "<obs>",
            grid_observation_to_text(example.observation),
            f"<action> {example.action.direction}",
            "<next_obs>",
            grid_observation_to_text(example.next_observation),
            f"<reward> {example.reward:g}",
            f"<terminated> {str(example.terminated).lower()}",
            f"<truncated> {str(example.truncated).lower()}",
        )
    )


def language_modeling_examples_from_grid_experience(
    examples: Sequence[GridExperienceTransition],
) -> list[LanguageModelingExample]:
    if not examples:
        raise ValueError("examples must not be empty")
    return [LanguageModelingExample(grid_experience_transition_to_text(example)) for example in examples]
