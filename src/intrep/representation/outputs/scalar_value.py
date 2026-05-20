from __future__ import annotations

import torch
from torch import nn


class ScalarTanhValueHead(nn.Module):
    def __init__(self, *, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, position_embedding: torch.Tensor) -> torch.Tensor:
        return self.scorer(position_embedding).squeeze(-1)
