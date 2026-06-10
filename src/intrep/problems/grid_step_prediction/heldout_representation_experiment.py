"""Temporary 2x2 held-out generalization experiment.

Crosses two factors suspected of blocking held-out agent-cell generalization:

- input: learned absolute position embeddings (current) vs explicit
  normalized row/col coordinate channels (no learned position embedding)
- output: absolute next-cell-id classification (current) vs relative-move
  classification (5 delta classes, decoded back to a next cell)

Remove this module after recording results in
`issues/grid-world-heldout-generalization.md`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from intrep.core.training_utils import build_adamw, clip_gradients
from intrep.domains.grid.encoding import grid_action_to_id, grid_observation_to_tensor, grid_position_to_cell_id
from intrep.domains.grid.layers import GridObservationInputLayer
from intrep.domains.grid.world import (
    GridExperienceTransition,
    GridWorldState,
    Position,
    generate_grid_world_transition_table,
)
from intrep.problems.grid_step_prediction.dataset import split_grid_transitions_by_agent_cell
from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.domains.grid.world import GRID_ACTIONS

MOVE_DELTAS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class _CoordinateGridInputLayer(nn.Module):
    """Grid input with explicit row/col coordinate channels instead of a
    learned absolute position embedding."""

    def __init__(self, *, height: int, width: int, embedding_dim: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.cell_projection = nn.Linear(5, embedding_dim)
        rows = torch.arange(height, dtype=torch.float32).reshape(height, 1).expand(height, width)
        cols = torch.arange(width, dtype=torch.float32).reshape(1, width).expand(height, width)
        coordinates = torch.stack(
            (
                rows / max(height - 1, 1),
                cols / max(width - 1, 1),
            )
        )
        self.register_buffer("coordinates", coordinates)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.size(0)
        coordinates = self.coordinates.unsqueeze(0).expand(batch_size, -1, -1, -1)
        channels = torch.cat((observations, coordinates), dim=1)
        cells = channels.permute(0, 2, 3, 1).reshape(batch_size, self.height * self.width, 5)
        return self.cell_projection(cells)


class _ExperimentModel(nn.Module):
    def __init__(
        self,
        *,
        height: int,
        width: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        coordinate_input: bool,
        relative_output: bool,
    ) -> None:
        super().__init__()
        if coordinate_input:
            self.grid_input: nn.Module = _CoordinateGridInputLayer(
                height=height, width=width, embedding_dim=embedding_dim
            )
        else:
            self.grid_input = GridObservationInputLayer(
                height=height, width=width, embedding_dim=embedding_dim
            )
        self.action_embedding = nn.Embedding(len(GRID_ACTIONS), embedding_dim)
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        target_classes = len(MOVE_DELTAS) if relative_output else height * width
        self.target_output = nn.Linear(embedding_dim, target_classes)
        self.reward_output = nn.Linear(embedding_dim, 3)
        self.terminated_output = nn.Linear(embedding_dim, 2)

    def forward(self, observations: torch.Tensor, action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_embeddings = self.grid_input(observations)
        action_embeddings = self.action_embedding(action_ids).unsqueeze(1)
        hidden = self.core(torch.cat((grid_embeddings, action_embeddings), dim=1), causal=False)
        pooled = hidden[:, -1, :]
        return (
            self.target_output(pooled),
            self.reward_output(pooled),
            self.terminated_output(pooled),
        )


def _move_delta_id(example: GridExperienceTransition) -> int:
    delta = (
        example.next_observation.agent.row - example.observation.agent.row,
        example.next_observation.agent.col - example.observation.agent.col,
    )
    return MOVE_DELTAS.index(delta)


def _reward_id(reward: float) -> int:
    return {-0.1: 0, -0.01: 1, 1.0: 2}[reward]


def _tensors(
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
    relative_output: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.stack([grid_observation_to_tensor(example.observation) for example in examples])
    action_ids = torch.tensor([grid_action_to_id(example.action) for example in examples], dtype=torch.long)
    if relative_output:
        targets = torch.tensor([_move_delta_id(example) for example in examples], dtype=torch.long)
    else:
        targets = torch.tensor(
            [grid_position_to_cell_id(example.next_observation.agent, width=width) for example in examples],
            dtype=torch.long,
        )
    rewards = torch.tensor([_reward_id(example.reward) for example in examples], dtype=torch.long)
    terminated = torch.tensor([int(example.terminated) for example in examples], dtype=torch.long)
    return observations, action_ids, targets, rewards, terminated


def _decoded_predictions(
    model: _ExperimentModel,
    examples: Sequence[GridExperienceTransition],
    *,
    width: int,
    relative_output: bool,
) -> list[dict[str, object]]:
    observations, action_ids, _, _, _ = _tensors(examples, width=width, relative_output=relative_output)
    model.eval()
    with torch.no_grad():
        target_logits, _, _ = model(observations, action_ids)
    model.train()
    predicted = target_logits.argmax(dim=1).tolist()
    records = []
    for example, prediction in zip(examples, predicted):
        if relative_output:
            delta = MOVE_DELTAS[prediction]
            decoded = Position(
                row=example.observation.agent.row + delta[0],
                col=example.observation.agent.col + delta[1],
            )
        else:
            decoded = Position(row=prediction // width, col=prediction % width)
        records.append(
            {
                "action": example.action.direction,
                "agent": asdict(example.observation.agent),
                "true_next": asdict(example.next_observation.agent),
                "predicted_next": asdict(decoded),
                "stays": example.next_observation.agent == example.observation.agent,
                "correct": decoded == example.next_observation.agent,
            }
        )
    return records


def _accuracy(records: Sequence[dict[str, object]]) -> float:
    return sum(record["correct"] for record in records) / len(records)


def _train_one(
    train_examples: Sequence[GridExperienceTransition],
    eval_examples: Sequence[GridExperienceTransition],
    *,
    state: GridWorldState,
    seed: int,
    coordinate_input: bool,
    relative_output: bool,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = _ExperimentModel(
        height=state.height,
        width=state.width,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        coordinate_input=coordinate_input,
        relative_output=relative_output,
    )
    optimizer = build_adamw(model, learning_rate=learning_rate, weight_decay=0.01)
    observations, action_ids, targets, rewards, terminated = _tensors(
        train_examples, width=state.width, relative_output=relative_output
    )
    generator = torch.Generator().manual_seed(seed)
    for _ in range(max_steps):
        indices = torch.randperm(len(train_examples), generator=generator)[:batch_size]
        optimizer.zero_grad(set_to_none=True)
        target_logits, reward_logits, terminated_logits = model(observations[indices], action_ids[indices])
        loss = (
            nn.functional.cross_entropy(target_logits, targets[indices])
            + nn.functional.cross_entropy(reward_logits, rewards[indices])
            + nn.functional.cross_entropy(terminated_logits, terminated[indices])
        )
        loss.backward()
        clip_gradients(model, 1.0)
        optimizer.step()
    eval_predictions = _decoded_predictions(
        model, eval_examples, width=state.width, relative_output=relative_output
    )
    return {
        "train_next_cell_accuracy": _accuracy(
            _decoded_predictions(model, train_examples, width=state.width, relative_output=relative_output)
        ),
        "eval_next_cell_accuracy": _accuracy(eval_predictions),
        "eval_predictions": eval_predictions,
    }


def _self_check(state: GridWorldState) -> None:
    """Validate the relative-target derivation against the world transition
    semantics before trusting any experiment numbers."""
    for example in generate_grid_world_transition_table(state):
        delta = MOVE_DELTAS[_move_delta_id(example)]
        decoded = Position(
            row=example.observation.agent.row + delta[0],
            col=example.observation.agent.col + delta[1],
        )
        assert decoded == example.next_observation.agent
        if example.next_observation.blocked:
            assert delta == (0, 0)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="2x2 held-out generalization experiment.")
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 32, 33])
    parser.add_argument("--grid-width", type=int, default=3)
    parser.add_argument("--grid-height", type=int, default=2)
    parser.add_argument(
        "--held-out-cell",
        type=int,
        nargs=2,
        action="append",
        metavar=("ROW", "COL"),
        help="Restrict the sweep to these held-out cells (default: every valid cell).",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    args = parser.parse_args(argv)

    goal = Position(row=args.grid_height - 1, col=args.grid_width - 1)
    wall = Position(row=1, col=1)
    walls = frozenset({wall}) if args.grid_height > 1 and args.grid_width > 1 and wall != goal else frozenset()
    state = GridWorldState(
        width=args.grid_width,
        height=args.grid_height,
        agent=Position(row=0, col=0),
        goal=goal,
        walls=walls,
    )
    _self_check(state)
    examples = generate_grid_world_transition_table(state)
    agent_cells = sorted(
        {example.observation.agent for example in examples},
        key=lambda position: (position.row, position.col),
    )
    if args.held_out_cell:
        requested = {Position(row=row, col=col) for row, col in args.held_out_cell}
        unknown = requested - set(agent_cells)
        if unknown:
            raise ValueError(f"held-out cells are not valid agent cells: {sorted((p.row, p.col) for p in unknown)}")
        agent_cells = [cell for cell in agent_cells if cell in requested]

    runs = []
    for coordinate_input in (False, True):
        for relative_output in (False, True):
            for held_out_cell in agent_cells:
                train_examples, eval_examples = split_grid_transitions_by_agent_cell(
                    examples, held_out_cells=[held_out_cell]
                )
                for seed in args.seeds:
                    metrics = _train_one(
                        train_examples,
                        eval_examples,
                        state=state,
                        seed=seed,
                        coordinate_input=coordinate_input,
                        relative_output=relative_output,
                        max_steps=args.max_steps,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        embedding_dim=args.embedding_dim,
                        num_heads=args.num_heads,
                        hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers,
                    )
                    runs.append(
                        {
                            "coordinate_input": coordinate_input,
                            "relative_output": relative_output,
                            "held_out_cell": asdict(held_out_cell),
                            "seed": seed,
                            **metrics,
                        }
                    )
            done = [run for run in runs if run["coordinate_input"] == coordinate_input and run["relative_output"] == relative_output]
            train_mean = sum(run["train_next_cell_accuracy"] for run in done) / len(done)
            eval_mean = sum(run["eval_next_cell_accuracy"] for run in done) / len(done)
            print(
                f"coordinate_input={coordinate_input} relative_output={relative_output}"
                f" runs={len(done)} train_mean={train_mean:.4f} eval_mean={eval_mean:.4f}"
            )

    payload = {
        "schema_version": "intrep.grid_step_heldout_representation_experiment.v1",
        "world": {
            "kind": "grid_world",
            "width": state.width,
            "height": state.height,
            "goal": asdict(state.goal),
            "walls": [asdict(wall) for wall in sorted(state.walls, key=lambda position: (position.row, position.col))],
        },
        "seeds": list(args.seeds),
        "model": {
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        },
        "optimization": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "runs": runs,
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
