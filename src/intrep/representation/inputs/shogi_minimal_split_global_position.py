from __future__ import annotations

import torch
from torch import nn

from intrep.representation.inputs.shogi_position_features.position_minimal_split_global import (
    SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT,
    SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT,
    SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT,
    SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_schema import SHOGI_POSITION_FEATURE_VOCAB_SIZE


class ShogiMinimalSplitGlobalPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.feature_embedding = nn.Embedding(SHOGI_POSITION_FEATURE_VOCAB_SIZE, embedding_dim)
        self.global_slot_embedding = nn.Embedding(SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT, embedding_dim)
        self.square_slot_embedding = nn.Embedding(SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT, embedding_dim)
        self.global_norm = nn.LayerNorm(embedding_dim)
        self.square_norm = nn.LayerNorm(embedding_dim)

    def forward(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        return torch.cat(
            (
                self._global_embeddings(position_features),
                self._square_embeddings(position_features),
            ),
            dim=1,
        )

    def _global_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        global_feature_ids = position_features.global_feature_ids
        slots = torch.arange(SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT, device=global_feature_ids.device).unsqueeze(0)
        return self.global_norm(self.feature_embedding(global_feature_ids) + self.global_slot_embedding(slots))

    def _square_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        square_feature_ids = position_features.square_feature_ids.squeeze(-1)
        square_slots = torch.arange(SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT, device=square_feature_ids.device).unsqueeze(
            0
        )
        return self.square_norm(self.feature_embedding(square_feature_ids) + self.square_slot_embedding(square_slots))


class ShogiMinimalSplitGlobalPositionAttentionLogitBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.square_relation_bias = nn.Embedding(17 * 17, 1)
        nn.init.zeros_(self.square_relation_bias.weight)
        self.register_buffer("square_relation_ids", _square_geometry_relation_ids(), persistent=False)

    def forward(self, position_features: ShogiPositionFeatures, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        if embeddings.size(1) != SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT:
            raise ValueError("embeddings must use the minimal-split-global shogi position length")
        bias = torch.zeros(
            (SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        square_bias = self.square_relation_bias(self.square_relation_ids.to(embeddings.device)).squeeze(-1)
        square_start = SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET
        square_end = square_start + SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT
        bias[square_start:square_end, square_start:square_end] = square_bias.to(dtype=embeddings.dtype)
        return bias.unsqueeze(0).expand(embeddings.size(0), -1, -1)


class ShogiMinimalSplitGlobalPositionEncoder(nn.Module):
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


def _square_geometry_relation_ids() -> torch.Tensor:
    relation_ids = torch.empty(
        (SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT, SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT),
        dtype=torch.long,
    )
    for from_square in range(SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT):
        from_file = from_square % 9
        from_rank = from_square // 9
        for to_square in range(SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT):
            to_file = to_square % 9
            to_rank = to_square // 9
            file_delta = to_file - from_file
            rank_delta = to_rank - from_rank
            relation_ids[from_square, to_square] = (rank_delta + 8) * 17 + file_delta + 8
    return relation_ids
