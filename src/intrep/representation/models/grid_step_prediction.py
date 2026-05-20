from __future__ import annotations

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.worlds.grid.layers import GridObservationInputLayer
from intrep.worlds.grid.world import GRID_ACTIONS


class GridStepPredictionModel(nn.Module):
    """Task model for grid transition prediction."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        core: SharedTransformerCore | None = None,
    ) -> None:
        super().__init__()
        self.grid_input = GridObservationInputLayer(height=height, width=width, embedding_dim=embedding_dim)
        self.action_embedding = nn.Embedding(len(GRID_ACTIONS), embedding_dim)
        self.core = core or SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.next_cell_output = nn.Linear(embedding_dim, height * width)
        self.reward_output = nn.Linear(embedding_dim, 3)
        self.terminated_output = nn.Linear(embedding_dim, 2)

    def forward(self, observations: torch.Tensor, action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_ids.ndim != 1:
            raise ValueError("action_ids must have shape [batch]")
        grid_embeddings = self.grid_input(observations)
        action_embeddings = self.action_embedding(action_ids).unsqueeze(1)
        hidden = self.core(torch.cat((grid_embeddings, action_embeddings), dim=1), causal=False)
        pooled = hidden[:, -1, :]
        return (
            self.next_cell_output(pooled),
            self.reward_output(pooled),
            self.terminated_output(pooled),
        )
