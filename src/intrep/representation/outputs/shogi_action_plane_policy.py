from __future__ import annotations

import torch
from torch import nn


class ShogiActionPlanePolicyHead(nn.Module):
    def __init__(self, *, embedding_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, position_embedding: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(position_embedding)
        return logits.masked_fill(~legal_action_mask, torch.finfo(logits.dtype).min)
