from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    def moved(self, delta: "Position") -> "Position":
        return Position(row=self.row + delta.row, col=self.col + delta.col)


@dataclass(frozen=True)
class GridAction:
    direction: str


@dataclass(frozen=True)
class GridWorldState:
    width: int
    height: int
    agent: Position
    goal: Position
    walls: frozenset[Position] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "walls", frozenset(self.walls))
        if self.width <= 0 or self.height <= 0:
            raise ValueError("grid dimensions must be positive")
        for name, position in (("agent", self.agent), ("goal", self.goal)):
            if not self.contains(position):
                raise ValueError(f"{name} position is outside the grid")
            if position in self.walls:
                raise ValueError(f"{name} position overlaps a wall")
        for wall in self.walls:
            if not self.contains(wall):
                raise ValueError("wall position is outside the grid")

    def contains(self, position: Position) -> bool:
        return 0 <= position.row < self.height and 0 <= position.col < self.width

    def with_agent(self, agent: Position) -> "GridWorldState":
        return GridWorldState(
            width=self.width,
            height=self.height,
            agent=agent,
            goal=self.goal,
            walls=self.walls,
        )


@dataclass(frozen=True)
class GridObservation:
    text: str
    grid: tuple[str, ...]
    agent: Position
    reached_goal: bool
    last_action: str | None = None
    blocked: bool = False


@dataclass(frozen=True)
class GridTransitionExample:
    id: str
    state_before: GridWorldState
    action: GridAction
    next_observation: GridObservation
    state_after: GridWorldState
    source: str = "grid_world"


ACTION_DELTAS = {
    "up": Position(row=-1, col=0),
    "down": Position(row=1, col=0),
    "left": Position(row=0, col=-1),
    "right": Position(row=0, col=1),
    "stay": Position(row=0, col=0),
}


class GridWorld:
    def __init__(self, initial_state: GridWorldState | None = None) -> None:
        self._state = initial_state or default_grid_world_state()

    @property
    def hidden_state(self) -> GridWorldState:
        return self._state

    def observe(self) -> GridObservation:
        return observation_from_state(self._state)

    def step(self, action: GridAction | str) -> GridObservation:
        action = coerce_action(action)
        next_state, blocked = transition_state(self._state, action)
        self._state = next_state
        return observation_from_state(next_state, last_action=action.direction, blocked=blocked)


def default_grid_world_state() -> GridWorldState:
    return GridWorldState(
        width=4,
        height=3,
        agent=Position(row=0, col=0),
        goal=Position(row=2, col=3),
        walls=frozenset({Position(row=1, col=1)}),
    )


def coerce_action(action: GridAction | str) -> GridAction:
    if isinstance(action, GridAction):
        return action
    return GridAction(direction=action)


def transition_state(state: GridWorldState, action: GridAction | str) -> tuple[GridWorldState, bool]:
    action = coerce_action(action)
    if action.direction not in ACTION_DELTAS:
        raise ValueError(f"unknown grid action: {action.direction}")

    candidate = state.agent.moved(ACTION_DELTAS[action.direction])
    blocked = not state.contains(candidate) or candidate in state.walls
    if blocked:
        return state, True
    return state.with_agent(candidate), False


def observation_from_state(
    state: GridWorldState,
    last_action: str | None = None,
    blocked: bool = False,
) -> GridObservation:
    reached_goal = state.agent == state.goal
    rows = []
    for row in range(state.height):
        cells = []
        for col in range(state.width):
            position = Position(row=row, col=col)
            if position == state.agent:
                cells.append("*" if reached_goal else "A")
            elif position in state.walls:
                cells.append("#")
            else:
                cells.append(".")
        rows.append("".join(cells))

    action_text = "initial" if last_action is None else f"after {last_action}"
    status_text = "blocked" if blocked else "moved"
    goal_text = "reached goal" if reached_goal else "goal not reached"
    text = f"{action_text}; agent at ({state.agent.row}, {state.agent.col}); {status_text}; {goal_text}"

    return GridObservation(
        text=text,
        grid=tuple(rows),
        agent=state.agent,
        reached_goal=reached_goal,
        last_action=last_action,
        blocked=blocked,
    )


def generate_grid_world_corpus(
    actions: Sequence[GridAction | str] | None = None,
    initial_state: GridWorldState | None = None,
) -> list[GridTransitionExample]:
    action_sequence = actions if actions is not None else (
        "right",
        "down",
        "down",
        "right",
        "right",
        "up",
        "left",
        "stay",
    )
    state = initial_state or default_grid_world_state()
    examples = []

    for index, action_input in enumerate(action_sequence, start=1):
        action = coerce_action(action_input)
        state_before = state
        state_after, blocked = transition_state(state_before, action)
        next_observation = observation_from_state(
            state_after,
            last_action=action.direction,
            blocked=blocked,
        )
        examples.append(
            GridTransitionExample(
                id=f"grid_case_{index}",
                state_before=state_before,
                action=action,
                next_observation=next_observation,
                state_after=state_after,
            )
        )
        state = state_after

    return examples
