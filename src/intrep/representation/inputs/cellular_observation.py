from __future__ import annotations

import torch
from torch import nn


class CellularObservationInputLayer(nn.Module):
    """One token per cell from a binary cellular observation."""

    def __init__(self, *, height: int, width: int, embedding_dim: int) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.cell_projection = nn.Linear(1, embedding_dim)
        self.position_embedding = nn.Embedding(height * width, embedding_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError("observations must have shape [batch, 1, height, width]")
        if observations.size(1) != 1:
            raise ValueError("cellular observations must have 1 channel")
        if observations.size(2) != self.height or observations.size(3) != self.width:
            raise ValueError("observation size does not match the input layer")
        batch_size = observations.size(0)
        cells = observations.permute(0, 2, 3, 1).reshape(batch_size, self.height * self.width, 1)
        positions = torch.arange(self.height * self.width, device=observations.device).unsqueeze(0)
        return self.cell_projection(cells) + self.position_embedding(positions)
