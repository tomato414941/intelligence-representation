from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import replace

import numpy as np
import torch

from intrep.experience.multimodal.records import MultimodalEpisode
from intrep.representation.inputs.multimodal_observation import MultimodalObservation
from intrep.worlds.gridworld.world import (
    GRID_ACTIONS,
    GridWorld,
    GridWorldState,
    Position,
    transition_state,
)

SAMPLE_RATE = 16000
AUDIO_SAMPLES = 512
COLORS = ("red", "blue")
PALETTE = {"floor": (0.08, 0.10, 0.13), "wall": (0.45, 0.48, 0.53),
           "agent": (1.0, 1.0, 1.0), "red": (0.95, 0.15, 0.18), "blue": (0.12, 0.40, 0.95)}
INSTRUCTIONS = ("Find the {color} marker.", "Go to the {color} marker.", "Reach the {color} marker.")


def tone(frequency: float, *, samples: int = AUDIO_SAMPLES, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    time = torch.arange(samples, dtype=torch.float32) / sample_rate
    return 0.6 * torch.sin(2 * torch.pi * frequency * time) * torch.hann_window(samples, periodic=False)


class MultimodalNavigationWorld:
    """Grounded integration scenario around the existing GridWorld dynamics.

    Images reveal the layout. Language identifies a marker. Sound identifies
    control mode. Those two instructions are absent from ordinary later frames.
    """

    def __init__(self, seed: int, *, size: tuple[int, int] | None = None, horizon: int = 6) -> None:
        if horizon < 2:
            raise ValueError("horizon must be at least two")
        self.seed, self.horizon, self.time = seed, horizon, 0
        self.rng = np.random.default_rng(seed)
        height, width = size or (int(self.rng.integers(3, 7)), int(self.rng.integers(3, 7)))
        if min(height, width) < 3:
            raise ValueError("navigation requires dimensions of at least three")
        cells = [Position(row, col) for row in range(height) for col in range(width)]
        chosen = self.rng.choice(len(cells), 3, replace=False)
        agent, red, blue = [cells[int(index)] for index in chosen]
        self.markers = (red, blue)
        self.target = int(self.rng.integers(2))
        self.reversed_controls = bool(self.rng.integers(2))
        self.change_target_at = horizon // 2 if self.rng.random() < 0.35 else -1
        self.change_controls_at = horizon // 2 if self.rng.random() < 0.35 else -1
        self.template = str(self.rng.choice(INSTRUCTIONS))
        state = GridWorldState(width, height, agent, self.markers[self.target])
        self.world = GridWorld(state)
        # The identity includes geometry, not target or mode, to group related worlds in splits.
        geometry = [height, width, [agent.row, agent.col], [red.row, red.col], [blue.row, blue.col]]
        self.world_id = hashlib.sha256(json.dumps(geometry).encode()).hexdigest()

    def _direction(self, action: int) -> str:
        if not 0 <= action < len(GRID_ACTIONS):
            raise ValueError("unknown action")
        direction = GRID_ACTIONS[action]
        if self.reversed_controls:
            direction = {"up": "down", "down": "up", "left": "right", "right": "left", "stay": "stay"}[direction]
        return direction

    def _image(self) -> torch.Tensor:
        state = self.world.hidden_state
        image = torch.tensor(PALETTE["floor"]).expand(state.height, state.width, 3).clone()
        for position in state.walls:
            image[position.row, position.col] = torch.tensor(PALETTE["wall"])
        for color, position in zip(COLORS, self.markers):
            image[position.row, position.col] = torch.tensor(PALETTE[color])
        image[state.agent.row, state.agent.col] = torch.tensor(PALETTE["agent"])
        return image

    def observe(self) -> MultimodalObservation:
        if self.time != 0:
            raise ValueError("initial observe is only available before the first step")
        return MultimodalObservation(
            text=self.template.format(color=COLORS[self.target]), image=self._image(),
            audio=tone(1500 if self.reversed_controls else 500), sample_rate=SAMPLE_RATE,
        )

    def expert_action(self) -> int:
        """Teacher construction uses privileged state; actor inputs never do."""
        state = self.world.hidden_state
        distances = {state.goal: 0}
        queue = deque([state.goal])
        while queue:
            position = queue.popleft()
            for direction in GRID_ACTIONS[:4]:
                neighbor, _ = transition_state(state.with_agent(position), direction)
                if neighbor.agent not in distances:
                    distances[neighbor.agent] = distances[position] + 1
                    queue.append(neighbor.agent)
        scores = []
        for action in range(len(GRID_ACTIONS)):
            after, _ = transition_state(state, self._direction(action))
            scores.append(distances.get(after.agent, state.width * state.height))
        if state.agent == state.goal:
            return GRID_ACTIONS.index("stay")
        best = min(scores)
        return next(index for index, score in enumerate(scores) if score == best)

    def step(self, action: int) -> MultimodalObservation:
        if self.time >= self.horizon:
            raise ValueError("episode horizon already reached")
        before = self.world.hidden_state
        distance_before = abs(before.agent.row - before.goal.row) + abs(before.agent.col - before.goal.col)
        result = self.world.step(self._direction(action))
        self.time += 1
        after = self.world.hidden_state
        distance_after = abs(after.agent.row - after.goal.row) + abs(after.agent.col - after.goal.col)
        reward = 1.0 if result.terminated else 0.1 * (distance_before - distance_after)
        text = ""
        if self.time == self.change_target_at:
            self.target = 1 - self.target
            self.world = GridWorld(replace(after, goal=self.markers[self.target]))
            text = "Now " + self.template[0].lower() + self.template[1:].format(color=COLORS[self.target])
        control_changed = self.time == self.change_controls_at
        if control_changed:
            self.reversed_controls = not self.reversed_controls
        frequency = (1500 if self.reversed_controls else 500) if control_changed else (
            1000 if result.terminated else 250 if result.observation.blocked else 750)
        # A marker is a continuing target; reaching it does not terminate the recording.
        feedback = torch.tensor([reward, 0.0, float(self.time == self.horizon)])
        return MultimodalObservation(text, self._image(), tone(frequency), SAMPLE_RATE, action, feedback)


def generate_episode(seed: int, *, size: tuple[int, int] | None = None, horizon: int = 6) -> MultimodalEpisode:
    world = MultimodalNavigationWorld(seed, size=size, horizon=horizon)
    observations = [world.observe()]
    actions, teachers, answers = [], [], []
    for _ in range(horizon):
        teacher = world.expert_action()
        # Vary recorded behavior while retaining an explicit teacher label for policy learning.
        action = int(world.rng.integers(len(GRID_ACTIONS))) if world.rng.random() < 0.35 else teacher
        teachers.append(teacher)
        answers.append(COLORS[world.target])
        actions.append(action)
        observations.append(world.step(action))
    return MultimodalEpisode(f"navigation-{seed}", world.world_id, observations, actions, teachers, answers,
                            {"generator": "multimodal_navigation", "seed": seed, "horizon": horizon,
                             "teacher": "shortest_path_with_world_state", "audio": "generated_control_and_outcome_cues"})
