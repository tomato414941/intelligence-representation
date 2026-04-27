from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

from intrep.grid_world import (
    GridTransitionExample,
    generate_grid_world_corpus,
    observation_from_state,
)
from intrep.mixed_corpus import MixedDocument


GridObservation = str | Sequence[str] | Sequence[Sequence[object]]


@dataclass(frozen=True)
class GridTransition:
    text: str
    grid: GridObservation
    action: object
    next_grid: GridObservation
    next_text: str


@dataclass(frozen=True)
class GridEpisode:
    id: str
    transitions: Sequence[GridTransition]


def episode_to_mixed_documents(
    episode: object,
    *,
    episode_id: str | None = None,
) -> list[MixedDocument]:
    resolved_episode_id = _renderable_id(episode_id or _read_episode_id(episode))
    documents: list[MixedDocument] = []

    for step_index, transition in enumerate(_read_transitions(episode), start=1):
        step_id = f"{resolved_episode_id}_step_{step_index:03d}"
        text = _required_transition_value(
            transition,
            ("text", "observation_text", "current_text"),
        )
        grid = _required_transition_value(
            transition,
            ("grid", "observation_grid", "current_grid"),
        )
        action = _required_transition_value(transition, ("action_log", "action"))
        next_grid = _required_transition_value(
            transition,
            ("next_grid", "next_observation_grid"),
        )
        next_text = _required_transition_value(
            transition,
            ("next_text", "next_observation_text", "expected_text"),
        )

        documents.extend(
            [
                MixedDocument(
                    id=f"{step_id}_text",
                    modality="text",
                    content=str(text),
                ),
                MixedDocument(
                    id=f"{step_id}_grid",
                    modality="grid",
                    content=_render_grid_document(grid),
                ),
                MixedDocument(
                    id=f"{step_id}_action_log",
                    modality="action_log",
                    content=_render_action_log(action),
                ),
                MixedDocument(
                    id=f"{step_id}_next_grid",
                    modality="next_grid",
                    content=_render_next_grid_document(next_grid),
                ),
                MixedDocument(
                    id=f"{step_id}_next_text",
                    modality="next_text",
                    content=str(next_text),
                ),
            ]
        )

    return documents


def episodes_to_mixed_documents(episodes: Iterable[object]) -> list[MixedDocument]:
    documents: list[MixedDocument] = []
    for episode in episodes:
        documents.extend(episode_to_mixed_documents(episode))
    return documents


def grid_world_examples_to_mixed_documents(
    examples: Iterable[GridTransitionExample],
) -> list[MixedDocument]:
    documents: list[MixedDocument] = []
    for example in examples:
        before_observation = observation_from_state(example.state_before)
        documents.extend(
            episode_to_mixed_documents(
                GridEpisode(
                    id=example.id,
                    transitions=[
                        GridTransition(
                            text=before_observation.text,
                            grid=before_observation.grid,
                            action={"type": "move", "direction": example.action.direction},
                            next_grid=example.next_observation.grid,
                            next_text=example.next_observation.text,
                        )
                    ],
                )
            )
        )
    return documents


def build_grid_corpus(episodes: Iterable[object] | None = None) -> list[MixedDocument]:
    if episodes is None:
        return default_grid_documents()
    return episodes_to_mixed_documents(episodes)


def default_grid_episodes() -> list[GridEpisode]:
    return [
        GridEpisode(
            id="grid_episode_001",
            transitions=[
                GridTransition(
                    text="The agent starts left of the goal.",
                    grid=("#####", "#A.G#", "#####"),
                    action="move_east",
                    next_grid=("#####", "#.AG#", "#####"),
                    next_text="The agent moves one cell east and is now next to the goal.",
                ),
                GridTransition(
                    text="The agent is next to the goal.",
                    grid=("#####", "#.AG#", "#####"),
                    action="move_east",
                    next_grid=("#####", "#..A#", "#####"),
                    next_text="The agent reaches the goal cell.",
                ),
            ],
        ),
        GridEpisode(
            id="grid_episode_002",
            transitions=[
                GridTransition(
                    text="A wall blocks the agent's northward move.",
                    grid=("#####", "#A..#", "#####"),
                    action="move_north",
                    next_grid=("#####", "#A..#", "#####"),
                    next_text="The wall blocks the action, so the grid does not change.",
                ),
            ],
        ),
    ]


def default_grid_documents() -> list[MixedDocument]:
    return grid_world_examples_to_mixed_documents(generate_grid_world_corpus())


def build_default_grid_corpus() -> list[MixedDocument]:
    return default_grid_documents()


def _read_episode_id(episode: object) -> str:
    episode_id = _read_value(episode, ("id", "episode_id", "name"))
    if episode_id is None:
        raise ValueError("grid episode must expose id, episode_id, or name")
    return str(episode_id)


def _read_transitions(episode: object) -> Sequence[object]:
    transitions = _read_value(episode, ("transitions", "steps"))
    if transitions is None:
        raise ValueError("grid episode must expose transitions or steps")
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise ValueError("grid episode transitions must be a sequence")
    return transitions


def _required_transition_value(transition: object, names: tuple[str, ...]) -> object:
    value = _read_value(transition, names)
    if value is None:
        expected = ", ".join(names)
        raise ValueError(f"grid transition must expose one of: {expected}")
    return value


def _read_value(source: object, names: tuple[str, ...]) -> object | None:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _render_grid_document(grid: object) -> str:
    return f"<grid>\n{_render_grid(grid)}\n</grid>"


def _render_next_grid_document(grid: object) -> str:
    return f"<next_grid>\n{_render_grid(grid)}\n</next_grid>"


def _render_grid(grid: object) -> str:
    if isinstance(grid, str):
        return grid.strip("\n")
    if not isinstance(grid, Sequence):
        raise ValueError("grid observation must be a string or row sequence")

    rows: list[str] = []
    for row in grid:
        if isinstance(row, str):
            rows.append(row)
        elif isinstance(row, Sequence):
            rows.append("".join(str(cell) for cell in row))
        else:
            raise ValueError("grid observation rows must be strings or cell sequences")
    return "\n".join(rows)


def _render_action_log(action: object) -> str:
    if isinstance(action, str):
        rendered_action = action
    elif isinstance(action, Mapping):
        rendered_action = " ".join(
            f"{key}={value}"
            for key, value in sorted(action.items(), key=lambda item: str(item[0]))
        )
    else:
        action_type = _read_value(action, ("type", "name"))
        if action_type is None:
            rendered_action = str(action)
        else:
            parts = [f"type={action_type}"]
            for name in ("actor", "object", "target", "direction"):
                value = _read_value(action, (name,))
                if value is not None:
                    parts.append(f"{name}={value}")
            rendered_action = " ".join(parts)
    return f"<action> {rendered_action}"


def _renderable_id(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    rendered = rendered.strip("_")
    if not rendered:
        raise ValueError("grid episode id must contain at least one renderable character")
    return rendered
