from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.worlds.shogi.move_encoding import NO_DROP_PIECE_ID, NO_FROM_SQUARE_ID
from intrep.worlds.shogi.position_encoding import (
    BOARD_TOKEN_OFFSET,
    SHOGI_POSITION_TOKEN_COUNT,
    SHOGI_POSITION_VOCAB_SIZE,
)
from intrep.core.transformer_core import SharedTransformerCore


FROM_SQUARE_VOCAB_SIZE = NO_FROM_SQUARE_ID + 1
TO_SQUARE_VOCAB_SIZE = 81
PROMOTION_VOCAB_SIZE = 2
DROP_PIECE_VOCAB_SIZE = 8
SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER = "shared_transformer"
SHOGI_POLICY_VALUE_MODEL_DIRECT = "direct"
SHOGI_POLICY_VALUE_MODEL_NAMES = (
    SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER,
    SHOGI_POLICY_VALUE_MODEL_DIRECT,
)
SHOGI_POSITION_INPUT_MODULE_ID = "shogi_side_to_move_relative_in_check_attack_count_position_tokens"
SHOGI_CANDIDATE_MOVE_INPUT_MODULE_ID = "shogi_side_to_move_relative_candidate_moves"
SHOGI_SHARED_CORE_MODULE_ID = "shared_transformer_core"
SHOGI_POSITION_POOLING_MODULE_ID = "mean_position_pooling"
SHOGI_POLICY_HEAD_MODULE_ID = "candidate_policy_head"
SHOGI_VALUE_HEAD_MODULE_ID = "scalar_tanh_value_head"
SHOGI_DIRECT_POSITION_POOLING_MODULE_ID = "mean_direct_position_embedding"
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


def shogi_policy_value_model_spec(model: str) -> dict[str, object]:
    if model == SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER:
        return dict(SHOGI_POLICY_VALUE_MODEL_SPEC)
    if model == SHOGI_POLICY_VALUE_MODEL_DIRECT:
        return dict(SHOGI_DIRECT_POLICY_VALUE_MODEL_SPEC)
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
        self.position_embedding = nn.Embedding(SHOGI_POSITION_VOCAB_SIZE, embedding_dim)
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
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
        move_embedding = self.move_input(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        return self.policy_head(torch.cat((expanded_position, move_embedding), dim=-1), candidate_mask)

    def forward_policy_value(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
        move_embedding = self.move_input(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        logits = self.policy_head(torch.cat((expanded_position, move_embedding), dim=-1), candidate_mask)
        return logits, self.value_head(position_embedding)

    def predict_value(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
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


class ShogiPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(SHOGI_POSITION_VOCAB_SIZE, embedding_dim)
        self.position_embedding = nn.Embedding(SHOGI_POSITION_TOKEN_COUNT, embedding_dim)

    def forward(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(position_token_ids.size(1), device=position_token_ids.device).unsqueeze(0)
        return self.token_embedding(position_token_ids) + self.position_embedding(positions)


class SharedCoreShogiPolicyValueModel(nn.Module):
    def __init__(self, config: SharedCoreShogiPolicyValueModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedCoreShogiPolicyValueModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_input = ShogiPositionInputLayer(embedding_dim=embedding_dim)
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
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
        position_embedding = position_hidden.mean(dim=1)
        candidate_inputs = self.candidate_policy_inputs(position_hidden, position_embedding, candidate_move_features)
        return self.policy_head(candidate_inputs, candidate_mask)

    def forward_policy_value(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
        position_embedding = position_hidden.mean(dim=1)
        candidate_inputs = self.candidate_policy_inputs(position_hidden, position_embedding, candidate_move_features)
        logits = self.policy_head(candidate_inputs, candidate_mask)
        return logits, self.value_head(position_embedding)

    def predict_value(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
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


def _candidate_square_hidden(
    position_hidden: torch.Tensor,
    square_ids: torch.Tensor,
    *,
    zero_square_id: int | None = None,
) -> torch.Tensor:
    embedding_dim = position_hidden.size(-1)
    zero_mask = square_ids.eq(zero_square_id) if zero_square_id is not None else torch.zeros_like(square_ids).bool()
    safe_square_ids = square_ids.masked_fill(zero_mask, 0)
    token_indices = safe_square_ids + BOARD_TOKEN_OFFSET
    gather_indices = token_indices[..., None].expand(-1, -1, embedding_dim)
    square_hidden = position_hidden.gather(dim=1, index=gather_indices)
    return square_hidden.masked_fill(zero_mask[..., None], 0.0)
