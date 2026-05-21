from __future__ import annotations

import torch
from torch import nn

from intrep.representation.inputs.shogi_position_features.position_rich import (
    RICH_LINE_ELEMENT_OFFSET,
    PAIR_RELATION_COUNT,
    SHOGI_RICH_POSITION_ELEMENT_COUNT,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_RICH_POSITION_GLOBAL_SLOT_COUNT,
    SHOGI_RICH_POSITION_LINE_FEATURE_COUNT,
    SHOGI_RICH_POSITION_LINE_SLOT_COUNT,
    SHOGI_RICH_POSITION_PIECE_FEATURE_COUNT,
    SHOGI_POSITION_SQUARE_COUNT,
    SHOGI_RICH_POSITION_SQUARE_FEATURE_COUNT,
    SHOGI_RICH_POSITION_SQUARE_SLOT_COUNT,
    RICH_SQUARE_ELEMENT_OFFSET,
    ShogiPositionFeatures,
    squares_for_line_index,
)


class ShogiRichPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.feature_embedding = nn.Embedding(SHOGI_POSITION_FEATURE_VOCAB_SIZE, embedding_dim)
        self.global_element_embedding = nn.Embedding(SHOGI_RICH_POSITION_GLOBAL_SLOT_COUNT, embedding_dim)
        self.square_position_embedding = nn.Embedding(SHOGI_RICH_POSITION_SQUARE_SLOT_COUNT, embedding_dim)
        self.square_feature_embedding = nn.Embedding(SHOGI_RICH_POSITION_SQUARE_FEATURE_COUNT, embedding_dim)
        self.piece_feature_embedding = nn.Embedding(SHOGI_RICH_POSITION_PIECE_FEATURE_COUNT, embedding_dim)
        self.line_feature_embedding = nn.Embedding(SHOGI_RICH_POSITION_LINE_FEATURE_COUNT, embedding_dim)
        self.line_element_embedding = nn.Embedding(SHOGI_RICH_POSITION_LINE_SLOT_COUNT, embedding_dim)
        self.global_norm = nn.LayerNorm(embedding_dim)
        self.square_norm = nn.LayerNorm(embedding_dim)
        self.piece_norm = nn.LayerNorm(embedding_dim)
        self.line_norm = nn.LayerNorm(embedding_dim)

    def forward(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        return torch.cat(
            (
                self._global_embeddings(position_features),
                self._square_embeddings(position_features),
                self._piece_embeddings(position_features),
                self._line_embeddings(position_features),
            ),
            dim=1,
        )

    def _global_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        global_feature_ids = position_features.global_feature_ids
        global_elements = torch.arange(SHOGI_RICH_POSITION_GLOBAL_SLOT_COUNT, device=global_feature_ids.device).unsqueeze(0)
        return self.global_norm(
            self.feature_embedding(global_feature_ids) + self.global_element_embedding(global_elements)
        )

    def _square_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        square_features = position_features.square_feature_ids
        square_feature_embeddings = self.feature_embedding(square_features)
        feature_slots = torch.arange(SHOGI_RICH_POSITION_SQUARE_FEATURE_COUNT, device=square_features.device)
        square_hidden = (
            square_feature_embeddings
            + self.square_feature_embedding(feature_slots).view(1, 1, SHOGI_RICH_POSITION_SQUARE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        square_positions = torch.arange(SHOGI_POSITION_SQUARE_COUNT, device=square_features.device).unsqueeze(0)
        return self.square_norm(square_hidden + self.square_position_embedding(square_positions))

    def _piece_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        piece_features = position_features.piece_feature_ids
        piece_feature_embeddings = self.feature_embedding(piece_features)
        feature_slots = torch.arange(SHOGI_RICH_POSITION_PIECE_FEATURE_COUNT, device=piece_features.device)
        piece_hidden = (
            piece_feature_embeddings
            + self.piece_feature_embedding(feature_slots).view(1, 1, SHOGI_RICH_POSITION_PIECE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        return self.piece_norm(piece_hidden)

    def _line_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        line_features = position_features.line_feature_ids
        line_feature_embeddings = self.feature_embedding(line_features)
        feature_slots = torch.arange(SHOGI_RICH_POSITION_LINE_FEATURE_COUNT, device=line_features.device)
        line_hidden = (
            line_feature_embeddings
            + self.line_feature_embedding(feature_slots).view(1, 1, SHOGI_RICH_POSITION_LINE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        line_elements = torch.arange(SHOGI_RICH_POSITION_LINE_SLOT_COUNT, device=line_features.device).unsqueeze(0)
        return self.line_norm(line_hidden + self.line_element_embedding(line_elements))


class ShogiRichPositionAttentionLogitBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relation_bias = nn.Embedding(17 * 17, 1)
        self.line_square_relation_bias = nn.Embedding(2, 1)
        self.pair_relation_bias = nn.Embedding(PAIR_RELATION_COUNT, 1)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.zeros_(self.line_square_relation_bias.weight)
        nn.init.zeros_(self.pair_relation_bias.weight)
        self.register_buffer("square_relation_ids", _square_geometry_relation_ids(), persistent=False)
        self.register_buffer("line_square_relation_ids", _line_square_relation_ids(), persistent=False)

    def forward(self, position_features: ShogiPositionFeatures, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        if embeddings.size(1) != SHOGI_RICH_POSITION_ELEMENT_COUNT:
            raise ValueError("embeddings must use the shogi position feature sequence length")
        bias = torch.zeros(
            (SHOGI_RICH_POSITION_ELEMENT_COUNT, SHOGI_RICH_POSITION_ELEMENT_COUNT),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        square_bias = self.relation_bias(self.square_relation_ids.to(embeddings.device)).squeeze(-1)
        square_start = RICH_SQUARE_ELEMENT_OFFSET
        square_end = RICH_SQUARE_ELEMENT_OFFSET + SHOGI_POSITION_SQUARE_COUNT
        bias[square_start:square_end, square_start:square_end] = square_bias.to(dtype=embeddings.dtype)
        line_start = RICH_LINE_ELEMENT_OFFSET
        line_end = RICH_LINE_ELEMENT_OFFSET + SHOGI_RICH_POSITION_LINE_SLOT_COUNT
        line_square_bias = self.line_square_relation_bias(self.line_square_relation_ids.to(embeddings.device)).squeeze(
            -1
        )
        line_square_bias = line_square_bias.to(dtype=embeddings.dtype)
        bias[line_start:line_end, square_start:square_end] = line_square_bias
        bias[square_start:square_end, line_start:line_end] = line_square_bias.transpose(0, 1)
        bias = bias.unsqueeze(0).expand(embeddings.size(0), -1, -1).clone()
        pair_relation_edges = position_features.pair_relation_edges.to(embeddings.device)
        if pair_relation_edges.relation_ids.numel() > 0:
            batch_indices = pair_relation_edges.batch_indices
            if batch_indices is None:
                batch_indices = torch.zeros_like(pair_relation_edges.relation_ids)
            pair_bias = self.pair_relation_bias(pair_relation_edges.relation_ids.long()).squeeze(-1).to(
                dtype=embeddings.dtype
            )
            bias.index_put_(
                (
                    batch_indices.long(),
                    pair_relation_edges.source_element_indices.long(),
                    pair_relation_edges.target_element_indices.long(),
                ),
                pair_bias,
                accumulate=True,
            )
        return bias


class ShogiRichPositionEncoder(nn.Module):
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
    relation_ids = torch.empty((SHOGI_POSITION_SQUARE_COUNT, SHOGI_POSITION_SQUARE_COUNT), dtype=torch.long)
    for from_square in range(SHOGI_POSITION_SQUARE_COUNT):
        from_file = from_square % 9
        from_rank = from_square // 9
        for to_square in range(SHOGI_POSITION_SQUARE_COUNT):
            to_file = to_square % 9
            to_rank = to_square // 9
            file_delta = to_file - from_file
            rank_delta = to_rank - from_rank
            relation_ids[from_square, to_square] = (rank_delta + 8) * 17 + file_delta + 8
    return relation_ids


def _line_square_relation_ids() -> torch.Tensor:
    relation_ids = torch.zeros((SHOGI_RICH_POSITION_LINE_SLOT_COUNT, SHOGI_POSITION_SQUARE_COUNT), dtype=torch.long)
    for line_index in range(SHOGI_RICH_POSITION_LINE_SLOT_COUNT):
        for square in squares_for_line_index(line_index):
            relation_ids[line_index, square] = 1
    return relation_ids
