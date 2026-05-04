from __future__ import annotations

import torch
from torch import nn


class SharedTransformerCore(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, embeddings: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        length = embeddings.size(1)
        # This boolean covers only the current bidirectional/causal cases.
        # Replace it with an explicit attention mask or pattern when prefix or
        # padding behavior becomes part of the main path.
        mask = torch.triu(
            torch.ones(length, length, device=embeddings.device, dtype=torch.bool),
            diagonal=1,
        ) if causal else None
        return self.encoder(embeddings, mask=mask)
