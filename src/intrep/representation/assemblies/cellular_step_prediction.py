from __future__ import annotations

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.cellular_observation import CellularObservationInputLayer


class CellularStepPredictionModel(nn.Module):
    """Predicts the next cellular observation per cell, from each cell's own
    token position. There is no action: the world updates by its rule alone.
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
        self.observation_input = CellularObservationInputLayer(
            height=height, width=width, embedding_dim=embedding_dim
        )
        self.core = core or SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.next_cell_output = nn.Linear(embedding_dim, 2)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.core(self.observation_input(observations), causal=False)
        return self.next_cell_output(hidden)
