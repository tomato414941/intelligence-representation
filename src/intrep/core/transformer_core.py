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
        self.num_heads = num_heads

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        causal: bool = False,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        length = embeddings.size(1)
        if attention_bias is not None and tuple(attention_bias.shape) not in (
            (length, length),
            (embeddings.size(0), length, length),
        ):
            raise ValueError("attention_bias must have shape [sequence, sequence] or [batch, sequence, sequence]")
        # This boolean covers only the current bidirectional/causal cases.
        # Replace it with an explicit attention mask or pattern when prefix or
        # padding behavior becomes part of the main path.
        if attention_bias is None:
            mask = torch.triu(
                torch.ones(length, length, device=embeddings.device, dtype=torch.bool),
                diagonal=1,
            ) if causal else None
        else:
            mask = attention_bias.to(device=embeddings.device, dtype=embeddings.dtype)
            if mask.ndim == 3:
                mask = mask.repeat_interleave(self.num_heads, dim=0)
            if causal:
                causal_mask = torch.triu(
                    torch.full((length, length), float("-inf"), device=embeddings.device, dtype=embeddings.dtype),
                    diagonal=1,
                )
                mask = mask + causal_mask
        if attention_bias is None:
            return self.encoder(embeddings, mask=mask)
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            return self.encoder(embeddings, mask=mask)
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
