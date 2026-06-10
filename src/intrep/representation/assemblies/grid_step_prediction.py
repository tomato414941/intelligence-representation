from __future__ import annotations

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.domains.grid.encoding import GRID_CELL_CLASSES
from intrep.domains.grid.layers import GridObservationInputLayer
from intrep.domains.grid.world import GRID_ACTIONS


class GridStepPredictionModel(nn.Module):
    """Task model for grid next-observation prediction.

    The next observation is predicted per cell, from each cell's own token
    position. The action token is optional so action-less lattice worlds can
    use the same head.
    """

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
        self.next_observation_output = nn.Linear(embedding_dim, len(GRID_CELL_CLASSES))
        self.reward_output = nn.Linear(embedding_dim, 3)
        self.terminated_output = nn.Linear(embedding_dim, 2)

    def forward(
        self,
        observations: torch.Tensor,
        action_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_embeddings = self.grid_input(observations)
        cell_count = grid_embeddings.size(1)
        if action_ids is not None:
            if action_ids.ndim != 1:
                raise ValueError("action_ids must have shape [batch]")
            tokens = torch.cat((grid_embeddings, self.action_embedding(action_ids).unsqueeze(1)), dim=1)
        else:
            tokens = grid_embeddings
        hidden = self.core(tokens, causal=False)
        cell_hidden = hidden[:, :cell_count, :]
        pooled = hidden[:, -1, :] if action_ids is not None else hidden.mean(dim=1)
        return (
            self.next_observation_output(cell_hidden),
            self.reward_output(pooled),
            self.terminated_output(pooled),
        )
