from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int = 64
    embedding_dim: int = 32
    num_heads: int = 4
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


class DecoderOnlyGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output = nn.Linear(config.embedding_dim, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        length = token_ids.size(1)
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        mask = torch.triu(
            torch.ones(length, length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        encoded = self.encoder(hidden, mask=mask)
        return self.output(encoded)
