from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.shogi.move_encoding import NO_DROP_PIECE_ID, NO_FROM_SQUARE_ID
from intrep.shogi.position_encoding import SHOGI_POSITION_TOKEN_COUNT, SHOGI_POSITION_VOCAB_SIZE
from intrep.core.transformer_core import SharedTransformerCore


FROM_SQUARE_VOCAB_SIZE = NO_FROM_SQUARE_ID + 1
TO_SQUARE_VOCAB_SIZE = 81
PROMOTION_VOCAB_SIZE = 2
DROP_PIECE_VOCAB_SIZE = 8


@dataclass(frozen=True)
class ShogiMoveChoiceModelConfig:
    embedding_dim: int = 32
    hidden_dim: int = 64


class ShogiMoveChoiceModel(nn.Module):
    def __init__(self, config: ShogiMoveChoiceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ShogiMoveChoiceModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_embedding = nn.Embedding(SHOGI_POSITION_VOCAB_SIZE, embedding_dim)
        self.from_square_embedding = nn.Embedding(FROM_SQUARE_VOCAB_SIZE, embedding_dim)
        self.to_square_embedding = nn.Embedding(TO_SQUARE_VOCAB_SIZE, embedding_dim)
        self.promotion_embedding = nn.Embedding(PROMOTION_VOCAB_SIZE, embedding_dim)
        self.drop_piece_embedding = nn.Embedding(DROP_PIECE_VOCAB_SIZE, embedding_dim)
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
        move_embedding = self.embed_candidate_moves(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        logits = self.scorer(torch.cat((expanded_position, move_embedding), dim=-1)).squeeze(-1)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)

    def forward_policy_value(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
        move_embedding = self.embed_candidate_moves(candidate_move_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        logits = self.scorer(torch.cat((expanded_position, move_embedding), dim=-1)).squeeze(-1)
        masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        values = self.value_head(position_embedding).squeeze(-1)
        return masked_logits, values

    def predict_value(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        position_embedding = self.position_embedding(position_token_ids).mean(dim=1)
        return self.value_head(position_embedding).squeeze(-1)

    def embed_candidate_moves(self, candidate_move_features: torch.Tensor) -> torch.Tensor:
        return (
            self.from_square_embedding(candidate_move_features[..., 0])
            + self.to_square_embedding(candidate_move_features[..., 1])
            + self.promotion_embedding(candidate_move_features[..., 2])
            + self.drop_piece_embedding(candidate_move_features[..., 3])
        )


@dataclass(frozen=True)
class SharedCoreShogiMoveChoiceModelConfig:
    embedding_dim: int = 32
    num_heads: int = 4
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0


class ShogiPositionInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(SHOGI_POSITION_VOCAB_SIZE, embedding_dim)
        self.position_embedding = nn.Embedding(SHOGI_POSITION_TOKEN_COUNT, embedding_dim)

    def forward(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(position_token_ids.size(1), device=position_token_ids.device).unsqueeze(0)
        return self.token_embedding(position_token_ids) + self.position_embedding(positions)


class SharedCoreShogiMoveChoiceModel(nn.Module):
    def __init__(self, config: SharedCoreShogiMoveChoiceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SharedCoreShogiMoveChoiceModelConfig()
        embedding_dim = self.config.embedding_dim
        self.position_input = ShogiPositionInputLayer(embedding_dim=embedding_dim)
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.move_model = ShogiMoveChoiceModel(
            ShogiMoveChoiceModelConfig(
                embedding_dim=embedding_dim,
                hidden_dim=self.config.hidden_dim,
            )
        )
        self.candidate_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 4, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, 1),
        )

    def forward(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
        position_embedding = position_hidden.mean(dim=1)
        logits = self.score_moves(position_hidden, position_embedding, candidate_move_features).squeeze(-1)
        return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)

    def forward_policy_value(
        self,
        position_token_ids: torch.Tensor,
        candidate_move_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
        position_embedding = position_hidden.mean(dim=1)
        logits = self.score_moves(position_hidden, position_embedding, candidate_move_features).squeeze(-1)
        masked_logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        values = self.move_model.value_head(position_embedding).squeeze(-1)
        return masked_logits, values

    def predict_value(self, position_token_ids: torch.Tensor) -> torch.Tensor:
        position_hidden = self.core(self.position_input(position_token_ids), causal=False)
        position_embedding = position_hidden.mean(dim=1)
        return self.move_model.value_head(position_embedding).squeeze(-1)

    def score_moves(
        self,
        position_hidden: torch.Tensor,
        position_embedding: torch.Tensor,
        candidate_move_features: torch.Tensor,
    ) -> torch.Tensor:
        move_embedding = self.move_model.embed_candidate_moves(candidate_move_features)
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
        return self.candidate_scorer(scorer_input)


def _candidate_square_hidden(
    position_hidden: torch.Tensor,
    square_ids: torch.Tensor,
    *,
    zero_square_id: int | None = None,
) -> torch.Tensor:
    embedding_dim = position_hidden.size(-1)
    zero_mask = square_ids.eq(zero_square_id) if zero_square_id is not None else torch.zeros_like(square_ids).bool()
    safe_square_ids = square_ids.masked_fill(zero_mask, 0)
    token_indices = safe_square_ids + 1
    gather_indices = token_indices[..., None].expand(-1, -1, embedding_dim)
    square_hidden = position_hidden.gather(dim=1, index=gather_indices)
    return square_hidden.masked_fill(zero_mask[..., None], 0.0)
