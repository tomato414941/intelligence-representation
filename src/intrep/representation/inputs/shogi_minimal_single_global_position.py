from __future__ import annotations

import torch
from torch import nn

from intrep.representation.inputs.shogi_square_geometry import (
    SHOGI_SQUARE_RELATIVE_POSITION_BUCKET_COUNT,
    shogi_square_relative_position_relation_ids,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT,
    SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FEATURE_COUNT,
    SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT,
    SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET,
)
from intrep.representation.inputs.shogi_position_features.position_schema import SHOGI_POSITION_FEATURE_VOCAB_SIZE


class ShogiMinimalSingleGlobalPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.feature_embedding = nn.Embedding(SHOGI_POSITION_FEATURE_VOCAB_SIZE, embedding_dim)
        self.global_feature_embedding = nn.Embedding(SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FEATURE_COUNT, embedding_dim)
        self.square_position_embedding = nn.Embedding(SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT, embedding_dim)
        self.global_norm = nn.LayerNorm(embedding_dim)
        self.square_norm = nn.LayerNorm(embedding_dim)

    def forward(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        return torch.cat(
            (
                self._global_embedding(position_features),
                self._square_embeddings(position_features),
            ),
            dim=1,
        )

    def _global_embedding(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        global_feature_ids = position_features.global_feature_ids
        feature_slots = torch.arange(SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FEATURE_COUNT, device=global_feature_ids.device)
        hidden = (
            self.feature_embedding(global_feature_ids)
            + self.global_feature_embedding(feature_slots).view(1, 1, SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FEATURE_COUNT, -1)
        ).sum(dim=2)
        return self.global_norm(hidden)

    def _square_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        square_feature_ids = position_features.square_feature_ids.squeeze(-1)
        square_positions = torch.arange(SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT, device=square_feature_ids.device).unsqueeze(
            0
        )
        return self.square_norm(self.feature_embedding(square_feature_ids) + self.square_position_embedding(square_positions))


class ShogiMinimalSingleGlobalPositionAttentionLogitBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square_relation_bias = nn.Embedding(SHOGI_SQUARE_RELATIVE_POSITION_BUCKET_COUNT, 1)
        nn.init.zeros_(self.square_relation_bias.weight)
        self.register_buffer(
            "square_relation_ids",
            shogi_square_relative_position_relation_ids(),
            persistent=False,
        )

    def forward(self, position_features: ShogiPositionFeatures, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        if embeddings.size(1) != SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT:
            raise ValueError("embeddings must use the minimal-single-global shogi position length")
        bias = torch.zeros(
            (SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        square_bias = self.square_relation_bias(self.square_relation_ids.to(embeddings.device)).squeeze(-1)
        square_start = SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET
        square_end = square_start + SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT
        bias[square_start:square_end, square_start:square_end] = square_bias.to(dtype=embeddings.dtype)
        return bias.unsqueeze(0).expand(embeddings.size(0), -1, -1)


class ShogiMinimalSingleGlobalPositionEncoder(nn.Module):
    def __init__(
        self,
        *,
        input_layer: nn.Module,
        core: nn.Module,
        attention_logit_bias: nn.Module,
    ) -> None:
        super().__init__()
        self.input = input_layer
        self.core = core
        self.attention_logit_bias = attention_logit_bias

    def forward(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_embeddings = self.input(position_features)
        return self.core(
            position_embeddings,
            causal=False,
            attention_logit_bias=self.attention_logit_bias(position_features, position_embeddings),
        )
