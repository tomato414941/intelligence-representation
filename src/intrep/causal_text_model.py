from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.transformer_core import SharedTransformerCore


@dataclass(frozen=True)
class CausalTextConfig:
    vocab_size: int
    context_length: int = 64
    embedding_dim: int = 32
    num_heads: int = 4
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


def build_causal_text_config(
    *,
    preset: str = "small",
    vocab_size: int,
    context_length: int,
    embedding_dim: int | None = None,
    num_heads: int | None = None,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
) -> CausalTextConfig:
    if preset not in TRANSFORMER_CORE_PRESETS:
        raise ValueError(f"unknown model preset: {preset}")
    preset_values = TRANSFORMER_CORE_PRESETS[preset]
    values: dict[str, Any] = {
        "vocab_size": vocab_size,
        "context_length": context_length,
        "embedding_dim": embedding_dim
        if embedding_dim is not None
        else preset_values["embedding_dim"],
        "num_heads": num_heads if num_heads is not None else preset_values["num_heads"],
        "hidden_dim": hidden_dim if hidden_dim is not None else preset_values["hidden_dim"],
        "num_layers": num_layers if num_layers is not None else preset_values["num_layers"],
        "dropout": dropout if dropout is not None else preset_values["dropout"],
    }
    config = CausalTextConfig(**values)
    validate_causal_text_config(config)
    return config


def validate_causal_text_config(config: CausalTextConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.context_length <= 0:
        raise ValueError("context_length must be positive")
    if config.embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be greater than or equal to 0.0 and less than 1.0")
    if config.embedding_dim % config.num_heads != 0:
        raise ValueError("embedding_dim must be divisible by num_heads")


def causal_text_config_to_dict(config: CausalTextConfig) -> dict[str, int | float]:
    return {
        "vocab_size": config.vocab_size,
        "context_length": config.context_length,
        "embedding_dim": config.embedding_dim,
        "num_heads": config.num_heads,
        "hidden_dim": config.hidden_dim,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
    }


class TokenOutputHead(nn.Module):
    def __init__(self, *, embedding_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        return self.output(hidden)


class CausalTextModel(nn.Module):
    def __init__(self, config: CausalTextConfig) -> None:
        super().__init__()
        validate_causal_text_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_dim)
        self.core = SharedTransformerCore(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.token_output = TokenOutputHead(
            embedding_dim=config.embedding_dim,
            vocab_size=config.vocab_size,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _validate_token_ids(token_ids, self.config)
        encoded = self.encode_tokens(token_ids)
        return self.token_logits(encoded)

    def embed_tokens(self, token_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        _validate_token_ids(token_ids, self.config)
        if position_offset < 0:
            raise ValueError("position_offset must be non-negative")
        length = token_ids.size(1)
        if position_offset + length > self.config.context_length:
            raise ValueError("token positions must not exceed context_length")
        positions = torch.arange(
            position_offset,
            position_offset + length,
            device=token_ids.device,
        ).unsqueeze(0)
        return self.token_embedding(token_ids) + self.position_embedding(positions)

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_embeddings(self.embed_tokens(token_ids), causal=True)

    def encode_embeddings(self, embeddings: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
        _validate_embeddings(embeddings, self.config)
        return self.core(embeddings, causal=causal)

    def token_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        _validate_hidden_states(hidden, self.config)
        return self.token_output(hidden)


def _validate_token_ids(token_ids: torch.Tensor, config: CausalTextConfig) -> None:
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be a rank-2 tensor with shape [batch, sequence]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must have dtype torch.long")
    if token_ids.size(1) > config.context_length:
        raise ValueError("token_ids sequence length must not exceed context_length")
    if token_ids.numel() == 0:
        raise ValueError("token_ids must not be empty")
    min_id = int(token_ids.min().item())
    max_id = int(token_ids.max().item())
    if min_id < 0 or max_id >= config.vocab_size:
        raise ValueError("token_ids values must be in the model vocabulary range")


def _validate_embeddings(embeddings: torch.Tensor, config: CausalTextConfig) -> None:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, sequence, hidden]")
    if not torch.is_floating_point(embeddings):
        raise ValueError("embeddings must be floating point")
    if embeddings.size(1) > config.context_length:
        raise ValueError("embeddings sequence length must not exceed context_length")
    if embeddings.size(2) != config.embedding_dim:
        raise ValueError("embeddings hidden size must match embedding_dim")
    if embeddings.numel() == 0:
        raise ValueError("embeddings must not be empty")


def _validate_hidden_states(hidden: torch.Tensor, config: CausalTextConfig) -> None:
    if hidden.ndim != 3:
        raise ValueError("hidden states must have shape [batch, sequence, hidden]")
    if not torch.is_floating_point(hidden):
        raise ValueError("hidden states must be floating point")
    if hidden.size(2) != config.embedding_dim:
        raise ValueError("hidden states hidden size must match embedding_dim")
    if hidden.numel() == 0:
        raise ValueError("hidden states must not be empty")
