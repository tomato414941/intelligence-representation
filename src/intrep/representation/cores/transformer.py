from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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
        self.layers = nn.ModuleList(
            [
                _SharedTransformerCoreBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_heads = num_heads

    def forward(
        self,
        embeddings: torch.Tensor,
        *,
        causal: bool = False,
        attention_logit_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        length = embeddings.size(1)
        if attention_logit_bias is not None and tuple(attention_logit_bias.shape) not in (
            (length, length),
            (embeddings.size(0), length, length),
        ):
            raise ValueError(
                "attention_logit_bias must have shape [sequence, sequence] or [batch, sequence, sequence]"
            )
        attention_mask = _attention_mask(
            embeddings,
            num_heads=self.num_heads,
            causal=causal,
            attention_logit_bias=attention_logit_bias,
        )
        hidden = embeddings
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_mask)
        return hidden


class _SharedTransformerCoreBlock(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_input = nn.Linear(embedding_dim, hidden_dim)
        self.feed_forward_output = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_after_attention = nn.LayerNorm(embedding_dim)
        self.norm_after_feed_forward = nn.LayerNorm(embedding_dim)

    def forward(self, hidden: torch.Tensor, *, attention_mask: torch.Tensor | None) -> torch.Tensor:
        attended, _weights = self.attention(hidden, hidden, hidden, attn_mask=attention_mask, need_weights=False)
        hidden = self.norm_after_attention(hidden + self.dropout(attended))
        feed_forward = self.feed_forward_output(self.dropout(F.gelu(self.feed_forward_input(hidden))))
        return self.norm_after_feed_forward(hidden + self.dropout(feed_forward))


def _attention_mask(
    embeddings: torch.Tensor,
    *,
    num_heads: int,
    causal: bool,
    attention_logit_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    length = embeddings.size(1)
    mask = None
    if attention_logit_bias is not None:
        mask = attention_logit_bias.to(device=embeddings.device, dtype=embeddings.dtype)
        if mask.ndim == 3:
            mask = mask.repeat_interleave(num_heads, dim=0)
    if causal:
        causal_mask = torch.triu(
            torch.full((length, length), float("-inf"), device=embeddings.device, dtype=embeddings.dtype),
            diagonal=1,
        )
        mask = causal_mask if mask is None else mask + causal_mask
    return mask
