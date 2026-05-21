from __future__ import annotations

import torch
from torch import nn


class TextTokenInputLayer(nn.Module):
    def __init__(self, *, vocab_size: int, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.context_length = context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(context_length, embedding_dim)

    def forward(self, token_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        if position_offset + token_ids.size(1) > self.context_length:
            raise ValueError("token positions must not exceed context_length")
        positions = torch.arange(
            position_offset,
            position_offset + token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        return self.token_embedding(token_ids) + self.position_embedding(positions)
