from __future__ import annotations

import torch
from torch import nn


class TokenOutputHead(nn.Module):
    def __init__(self, *, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        return self.output(hidden)
