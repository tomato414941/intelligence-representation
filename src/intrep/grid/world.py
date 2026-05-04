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
class GridStepResult:
    observation: GridObservation
    reward: float
    terminated: bool
    truncated: bool = False
    info: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GridExperienceTransition:
    id: str
    observation: GridObservation
    action: GridAction
    reward: float
    next_observation: GridObservation
    terminated: bool
    truncated: bool = False
    source: str = "grid_world"


ACTION_DELTAS = {
    "up": Position(row=-1, col=0),
    "down": Position(row=1, col=0),
    "left": Position(row=0, col=-1),
    "right": Position(row=0, col=1),
    "stay": Position(row=0, col=0),
}

GRID_ACTIONS = tuple(ACTION_DELTAS.keys())


class GridWorld:
    def __init__(self, initial_state: GridWorldState | None = None) -> None:
        self._state = initial_state or default_grid_world_state()

    @property
    def hidden_state(self) -> GridWorldState:
        return self._state

    def observe(self) -> GridObservation:
        return observation_from_state(self._state)

    def step(self, action: GridAction | str) -> GridStepResult:
        action = coerce_action(action)
        next_state, blocked = transition_state(self._state, action)
        self._state = next_state
        observation = observation_from_state(next_state, last_action=action.direction, blocked=blocked)
        return step_result_from_observation(observation)


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
            elif position == state.goal:
                cells.append("G")
            elif position in state.walls:
                cells.append("#")
            else:
                cells.append(".")
        rows.append("".join(cells))

    action_text = "initial" if last_action is None else f"after {last_action}"
    status_text = "blocked" if blocked else "moved"
    goal_text = "reached goal" if reached_goal else "goal not reached"
    text = (
        f"{action_text}; agent at ({state.agent.row}, {state.agent.col}); "
        f"goal at ({state.goal.row}, {state.goal.col}); {status_text}; {goal_text}"
    )

    return GridObservation(
        text=text,
        grid=tuple(rows),
        agent=state.agent,
        reached_goal=reached_goal,
        last_action=last_action,
        blocked=blocked,
    )


def step_result_from_observation(observation: GridObservation) -> GridStepResult:
    reward = 1.0 if observation.reached_goal else -0.1 if observation.blocked else -0.01
    return GridStepResult(
        observation=observation,
        reward=reward,
        terminated=observation.reached_goal,
    )


def generate_grid_world_experience(
    actions: Sequence[GridAction | str] | None = None,
    initial_state: GridWorldState | None = None,
) -> list[GridExperienceTransition]:
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
        observation = observation_from_state(state)
        state_after, blocked = transition_state(state, action)
        next_observation = observation_from_state(
            state_after,
            last_action=action.direction,
            blocked=blocked,
        )
        step_result = step_result_from_observation(next_observation)
        examples.append(
            GridExperienceTransition(
                id=f"grid_case_{index}",
                observation=observation,
                action=action,
                reward=step_result.reward,
                next_observation=next_observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
            )
        )
        state = state_after

    return examples


def generate_grid_world_transition_table(
    state_template: GridWorldState | None = None,
) -> list[GridExperienceTransition]:
    template = state_template or default_grid_world_state()
    examples: list[GridExperienceTransition] = []
    index = 0
    for row in range(template.height):
        for col in range(template.width):
            agent = Position(row=row, col=col)
            if agent in template.walls:
                continue
            state = GridWorldState(
                width=template.width,
                height=template.height,
                agent=agent,
                goal=template.goal,
                walls=template.walls,
            )
            for action_direction in GRID_ACTIONS:
                action = GridAction(direction=action_direction)
                observation = observation_from_state(state)
                state_after, blocked = transition_state(state, action)
                next_observation = observation_from_state(
                    state_after,
                    last_action=action.direction,
                    blocked=blocked,
                )
                step_result = step_result_from_observation(next_observation)
                index += 1
                examples.append(
                    GridExperienceTransition(
                        id=f"grid_transition_{index}",
                        observation=observation,
                        action=action,
                        reward=step_result.reward,
                        next_observation=next_observation,
                        terminated=step_result.terminated,
                        truncated=step_result.truncated,
                    )
                )
    return examples

