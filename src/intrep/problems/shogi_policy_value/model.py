from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.worlds.shogi.move_encoding import NO_DROP_PIECE_ID, NO_FROM_SQUARE_ID
from intrep.worlds.shogi.position_encoding import (
    LINE_TOKEN_OFFSET,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_POSITION_GLOBAL_SLOT_COUNT,
    SHOGI_POSITION_PIECE_FEATURE_COUNT,
    SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT,
    SHOGI_POSITION_LINE_FEATURE_COUNT,
    SHOGI_POSITION_LINE_SLOT_COUNT,
    SHOGI_POSITION_SQUARE_COUNT,
    SHOGI_POSITION_SQUARE_FEATURE_COUNT,
    SHOGI_POSITION_SQUARE_SLOT_COUNT,
    ShogiPositionFeatures,
    SQUARE_TOKEN_OFFSET,
    squares_for_line_index,
)
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT
from intrep.core.transformer_core import SharedTransformerCore


FROM_SQUARE_VOCAB_SIZE = NO_FROM_SQUARE_ID + 1
TO_SQUARE_VOCAB_SIZE = 81
PROMOTION_VOCAB_SIZE = 2
DROP_PIECE_VOCAB_SIZE = 8
SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER = "shared_transformer"
SHOGI_POLICY_VALUE_MODEL_DIRECT = "direct"
SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER = "policy_plane_shared_transformer"
SHOGI_POLICY_VALUE_MODEL_NAMES = (
    SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER,
    SHOGI_POLICY_VALUE_MODEL_DIRECT,
    SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
)
SHOGI_POSITION_INPUT_MODULE_ID = "shogi_global_square_drop_shadow_piece_line_state_position_features"
SHOGI_CANDIDATE_MOVE_INPUT_MODULE_ID = "shogi_side_to_move_relative_candidate_moves"
SHOGI_SHARED_CORE_MODULE_ID = "shared_transformer_core_with_shogi_position_geometry_bias"
SHOGI_POSITION_POOLING_MODULE_ID = "mean_position_pooling"
SHOGI_POLICY_HEAD_MODULE_ID = "candidate_policy_head"
SHOGI_POLICY_PLANE_HEAD_MODULE_ID = "policy_plane_head"
SHOGI_VALUE_HEAD_MODULE_ID = "scalar_tanh_value_head"
SHOGI_DIRECT_POSITION_POOLING_MODULE_ID = "mean_direct_position_feature_sequence_embedding"
SHOGI_POLICY_VALUE_MODEL_SPEC = {
    "position_input": SHOGI_POSITION_INPUT_MODULE_ID,
    "candidate_move_input": SHOGI_CANDIDATE_MOVE_INPUT_MODULE_ID,
    "core": SHOGI_SHARED_CORE_MODULE_ID,
    "position_pooling": SHOGI_POSITION_POOLING_MODULE_ID,
    "policy_head": SHOGI_POLICY_HEAD_MODULE_ID,
    "value_head": SHOGI_VALUE_HEAD_MODULE_ID,
}
SHOGI_DIRECT_POLICY_VALUE_MODEL_SPEC = {
    "position_input": SHOGI_POSITION_INPUT_MODULE_ID,
    "candidate_move_input": SHOGI_CANDIDATE_MOVE_INPUT_MODULE_ID,
    "core": None,
    "position_pooling": SHOGI_DIRECT_POSITION_POOLING_MODULE_ID,
    "policy_head": SHOGI_POLICY_HEAD_MODULE_ID,
    "value_head": SHOGI_VALUE_HEAD_MODULE_ID,
}
SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC = {
    "position_input": SHOGI_POSITION_INPUT_MODULE_ID,
    "candidate_move_input": None,
    "core": SHOGI_SHARED_CORE_MODULE_ID,
    "position_pooling": SHOGI_POSITION_POOLING_MODULE_ID,
    "policy_head": SHOGI_POLICY_PLANE_HEAD_MODULE_ID,
    "value_head": SHOGI_VALUE_HEAD_MODULE_ID,
}


def shogi_policy_value_model_spec(model: str) -> dict[str, object]:
    if model == SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER:
        return dict(SHOGI_POLICY_VALUE_MODEL_SPEC)
    if model == SHOGI_POLICY_VALUE_MODEL_DIRECT:
        return dict(SHOGI_DIRECT_POLICY_VALUE_MODEL_SPEC)
    if model == SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER:
        return dict(SHOGI_POLICY_PLANE_POLICY_VALUE_MODEL_SPEC)
    raise ValueError(f"unsupported shogi policy/value model: {model}")


@dataclass(frozen=True)
class DirectShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    hidden_dim: int = 1024


class DirectShogiPolicyValueModel(nn.Module):
    def __init__(self, config: DirectShogiPolicyValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or DirectShogiPolicyValueModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_input = ShogiPositionInputLayer(embedding_dim=embedding_dim)
        self.move_input = ShogiCandidateMoveInputLayer(embedding_dim=embedding_dim)
        self.policy_head = ShogiCandidateMovePolicyHead(
            input_dim=embedding_dim * 2,
            hidden_dim=self.config.hidden_dim,
        )
        self.value_head = ShogiValueHead(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = self.position_input(position_features).mean(dim=1)
        move_embedding = self.move_input(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        return self.policy_head(torch.cat((expanded_position, move_embedding), dim=-1), candidate_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embedding = self.position_input(position_features).mean(dim=1)
        move_embedding = self.move_input(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        logits = self.policy_head(torch.cat((expanded_position, move_embedding), dim=-1), candidate_mask)
        return logits, self.value_head(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_embedding = self.position_input(position_features).mean(dim=1)
        return self.value_head(position_embedding)


class ShogiCandidateMoveInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.from_square_embedding = nn.Embedding(FROM_SQUARE_VOCAB_SIZE, embedding_dim)
        self.to_square_embedding = nn.Embedding(TO_SQUARE_VOCAB_SIZE, embedding_dim)
        self.promotion_embedding = nn.Embedding(PROMOTION_VOCAB_SIZE, embedding_dim)
        self.drop_piece_embedding = nn.Embedding(DROP_PIECE_VOCAB_SIZE, embedding_dim)

    def forward(self, candidate_move_features: torch.Tensor) -> torch.Tensor:
        return (
            self.from_square_embedding(candidate_move_features[..., 0])
            + self.to_square_embedding(candidate_move_features[..., 1])
            + self.promotion_embedding(candidate_move_features[..., 2])
            + self.drop_piece_embedding(candidate_move_features[..., 3])
        )


class ShogiCandidateMovePolicyHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_inputs: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(candidate_inputs).squeeze(-1)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


class ShogiPolicyPlaneHead(nn.Module):
    def __init__(self, *, embedding_dim: int, hidden_dim: int, action_count: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, position_embedding: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(position_embedding)
        return logits.masked_fill(~legal_action_mask, torch.finfo(logits.dtype).min)


class ShogiValueHead(nn.Module):
    def __init__(self, *, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, position_embedding: torch.Tensor) -> torch.Tensor:
        return self.scorer(position_embedding).squeeze(-1)


@dataclass(frozen=True)
class SharedCoreShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyPlaneShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.0


class ShogiPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(SHOGI_POSITION_FEATURE_VOCAB_SIZE, embedding_dim)
        self.global_slot_embedding = nn.Embedding(SHOGI_POSITION_GLOBAL_SLOT_COUNT, embedding_dim)
        self.square_slot_embedding = nn.Embedding(SHOGI_POSITION_SQUARE_SLOT_COUNT, embedding_dim)
        self.square_feature_embedding = nn.Embedding(SHOGI_POSITION_SQUARE_FEATURE_COUNT, embedding_dim)
        self.piece_feature_embedding = nn.Embedding(SHOGI_POSITION_PIECE_FEATURE_COUNT, embedding_dim)
        self.line_feature_embedding = nn.Embedding(SHOGI_POSITION_LINE_FEATURE_COUNT, embedding_dim)
        self.line_slot_embedding = nn.Embedding(SHOGI_POSITION_LINE_SLOT_COUNT, embedding_dim)

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
        global_token_ids = position_features.global_feature_ids
        slots = torch.arange(SHOGI_POSITION_GLOBAL_SLOT_COUNT, device=global_token_ids.device).unsqueeze(0)
        return self.token_embedding(global_token_ids) + self.global_slot_embedding(slots)

    def _square_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        square_features = position_features.square_feature_ids
        square_feature_embeddings = self.token_embedding(square_features)
        feature_slots = torch.arange(SHOGI_POSITION_SQUARE_FEATURE_COUNT, device=square_features.device)
        square_hidden = (
            square_feature_embeddings
            + self.square_feature_embedding(feature_slots).view(1, 1, SHOGI_POSITION_SQUARE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        square_slots = torch.arange(SHOGI_POSITION_SQUARE_COUNT, device=square_features.device).unsqueeze(0)
        return square_hidden + self.square_slot_embedding(square_slots)

    def _piece_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        piece_features = position_features.piece_feature_ids
        piece_feature_embeddings = self.token_embedding(piece_features)
        feature_slots = torch.arange(SHOGI_POSITION_PIECE_FEATURE_COUNT, device=piece_features.device)
        piece_hidden = (
            piece_feature_embeddings
            + self.piece_feature_embedding(feature_slots).view(1, 1, SHOGI_POSITION_PIECE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        return piece_hidden

    def _line_embeddings(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        line_features = position_features.line_feature_ids
        line_feature_embeddings = self.token_embedding(line_features)
        feature_slots = torch.arange(SHOGI_POSITION_LINE_FEATURE_COUNT, device=line_features.device)
        line_hidden = (
            line_feature_embeddings
            + self.line_feature_embedding(feature_slots).view(1, 1, SHOGI_POSITION_LINE_FEATURE_COUNT, -1)
        ).sum(dim=2)
        slots = torch.arange(SHOGI_POSITION_LINE_SLOT_COUNT, device=line_features.device).unsqueeze(0)
        return line_hidden + self.line_slot_embedding(slots)


class ShogiPositionGeometryAttentionBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relation_bias = nn.Embedding(17 * 17, 1)
        self.line_square_relation_bias = nn.Embedding(2, 1)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.zeros_(self.line_square_relation_bias.weight)
        self.register_buffer("square_relation_ids", _square_geometry_relation_ids(), persistent=False)
        self.register_buffer("line_square_relation_ids", _line_square_relation_ids(), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        if embeddings.size(1) != SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT:
            raise ValueError("embeddings must use the shogi position feature sequence length")
        bias = torch.zeros(
            (SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT, SHOGI_POSITION_FEATURE_SEQUENCE_TOKEN_COUNT),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        square_bias = self.relation_bias(self.square_relation_ids.to(embeddings.device)).squeeze(-1)
        square_start = SQUARE_TOKEN_OFFSET
        square_end = SQUARE_TOKEN_OFFSET + SHOGI_POSITION_SQUARE_COUNT
        bias[square_start:square_end, square_start:square_end] = square_bias.to(dtype=embeddings.dtype)
        line_start = LINE_TOKEN_OFFSET
        line_end = LINE_TOKEN_OFFSET + SHOGI_POSITION_LINE_SLOT_COUNT
        line_square_bias = self.line_square_relation_bias(self.line_square_relation_ids.to(embeddings.device)).squeeze(
            -1
        )
        line_square_bias = line_square_bias.to(dtype=embeddings.dtype)
        bias[line_start:line_end, square_start:square_end] = line_square_bias
        bias[square_start:square_end, line_start:line_end] = line_square_bias.transpose(0, 1)
        return bias


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
    relation_ids = torch.zeros((SHOGI_POSITION_LINE_SLOT_COUNT, SHOGI_POSITION_SQUARE_COUNT), dtype=torch.long)
    for line_index in range(SHOGI_POSITION_LINE_SLOT_COUNT):
        for square in squares_for_line_index(line_index):
            relation_ids[line_index, square] = 1
    return relation_ids


class SharedCoreShogiPolicyValueModel(nn.Module):
    def __init__(self, config: SharedCoreShogiPolicyValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedCoreShogiPolicyValueModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_input = ShogiPositionInputLayer(embedding_dim=embedding_dim)
        self.position_attention_bias = ShogiPositionGeometryAttentionBias()
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.move_input = ShogiCandidateMoveInputLayer(embedding_dim=embedding_dim)
        self.policy_head = ShogiCandidateMovePolicyHead(
            input_dim=embedding_dim * 4,
            hidden_dim=self.config.hidden_dim,
        )
        self.value_head = ShogiValueHead(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        candidate_inputs = self.candidate_policy_inputs(position_hidden, position_embedding, candidate_move_features)
        return self.policy_head(candidate_inputs, candidate_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        candidate_inputs = self.candidate_policy_inputs(position_hidden, position_embedding, candidate_move_features)
        logits = self.policy_head(candidate_inputs, candidate_mask)
        return logits, self.value_head(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        return self.value_head(position_embedding)

    def candidate_policy_inputs(
        self,
        position_hidden: torch.Tensor,
        position_embedding: torch.Tensor,
        candidate_move_features: torch.Tensor,
    ) -> torch.Tensor:
        move_embedding = self.move_input(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        from_square_hidden = _candidate_square_hidden(
            position_hidden,
            candidate_move_features[..., 0],
            zero_square_id=NO_FROM_SQUARE_ID,
        )
        to_square_hidden = _candidate_square_hidden(position_hidden, candidate_move_features[..., 1])
        scorer_input = torch.cat(
            (expanded_position, move_embedding, from_square_hidden, to_square_hidden),
            dim=-1,
        )
        return scorer_input


class PolicyPlaneShogiPolicyValueModel(nn.Module):
    def __init__(self, config: PolicyPlaneShogiPolicyValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyPlaneShogiPolicyValueModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_input = ShogiPositionInputLayer(embedding_dim=embedding_dim)
        self.position_attention_bias = ShogiPositionGeometryAttentionBias()
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.policy_head = ShogiPolicyPlaneHead(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
            action_count=SHOGI_POLICY_PLANE_ACTION_COUNT,
        )
        self.value_head = ShogiValueHead(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(self, position_features: ShogiPositionFeatures, policy_plane_legal_mask: torch.Tensor) -> torch.Tensor:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        return self.policy_head(position_embedding, policy_plane_legal_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        policy_plane_legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        logits = self.policy_head(position_embedding, policy_plane_legal_mask)
        return logits, self.value_head(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_embeddings = self.position_input(position_features)
        position_hidden = self.core(
            position_embeddings,
            causal=False,
            attention_bias=self.position_attention_bias(position_embeddings),
        )
        position_embedding = position_hidden.mean(dim=1)
        return self.value_head(position_embedding)


def _candidate_square_hidden(
    position_hidden: torch.Tensor,
    square_ids: torch.Tensor,
    *,
    zero_square_id: int | None = None,
) -> torch.Tensor:
    embedding_dim = position_hidden.size(-1)
    zero_mask = square_ids.eq(zero_square_id) if zero_square_id is not None else torch.zeros_like(square_ids).bool()
    safe_square_ids = square_ids.masked_fill(zero_mask, 0)
    token_indices = safe_square_ids + SQUARE_TOKEN_OFFSET
    gather_indices = token_indices[..., None].expand(-1, -1, embedding_dim)
    square_hidden = position_hidden.gather(dim=1, index=gather_indices)
    return square_hidden.masked_fill(zero_mask[..., None], 0.0)
