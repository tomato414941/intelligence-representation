from __future__ import annotations

import random

from intrep.worlds.gridworld.world import GridWorldState, Position


def generate_random_grid_world_state(
    *,
    width: int,
    height: int,
    wall_count: int,
    seed: int,
) -> GridWorldState:
    """Generate a layout with randomly placed walls and goal.

    The agent position is a placeholder on a free cell; transition-table
    generation enumerates every free cell as the agent position anyway.
    """
    cell_count = width * height
    if wall_count < 0 or wall_count > cell_count - 2:
        raise ValueError("wall_count must leave room for the goal and the agent")
    cells = [Position(row=row, col=col) for row in range(height) for col in range(width)]
    sampled = random.Random(seed).sample(cells, wall_count + 2)
    return GridWorldState(
        width=width,
        height=height,
        agent=sampled[wall_count + 1],
        goal=sampled[wall_count],
        walls=frozenset(sampled[:wall_count]),
    )


def generate_grid_world_layouts(
    count: int,
    *,
    width: int,
    height: int,
    wall_count: int,
    seed: int,
) -> list[GridWorldState]:
    if count <= 0:
        raise ValueError("count must be positive")
    return [
        generate_random_grid_world_state(
            width=width,
            height=height,
            wall_count=wall_count,
            seed=seed + index,
        )
        for index in range(count)
    ]
