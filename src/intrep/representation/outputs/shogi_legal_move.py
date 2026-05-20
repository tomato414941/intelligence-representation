from __future__ import annotations

import torch
from torch import nn

from intrep.worlds.shogi.move_encoding import NO_FROM_SQUARE_ID
from intrep.worlds.shogi.position_encoding import SQUARE_ELEMENT_OFFSET


FROM_SQUARE_VOCAB_SIZE = NO_FROM_SQUARE_ID + 1
TO_SQUARE_VOCAB_SIZE = 81
PROMOTION_VOCAB_SIZE = 2
DROP_PIECE_VOCAB_SIZE = 8


class ShogiLegalMoveInputLayer(nn.Module):
    def __init__(self, *, embedding_dim: int) -> None:
        super().__init__()
        self.from_square_embedding = nn.Embedding(FROM_SQUARE_VOCAB_SIZE, embedding_dim)
        self.to_square_embedding = nn.Embedding(TO_SQUARE_VOCAB_SIZE, embedding_dim)
        self.promotion_embedding = nn.Embedding(PROMOTION_VOCAB_SIZE, embedding_dim)
        self.drop_piece_embedding = nn.Embedding(DROP_PIECE_VOCAB_SIZE, embedding_dim)

    def forward(self, legal_move_features: torch.Tensor) -> torch.Tensor:
        return (
            self.from_square_embedding(legal_move_features[..., 0])
            + self.to_square_embedding(legal_move_features[..., 1])
            + self.promotion_embedding(legal_move_features[..., 2])
            + self.drop_piece_embedding(legal_move_features[..., 3])
        )


class ShogiStateSummaryLegalMovePolicyHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_move_inputs: torch.Tensor, legal_move_mask: torch.Tensor) -> torch.Tensor:
        logits = self.scorer(state_move_inputs).squeeze(-1)
        return logits.masked_fill(~legal_move_mask, torch.finfo(logits.dtype).min)


class ShogiStateSummaryLegalMovePolicyOutput(nn.Module):
    def __init__(self, *, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.move_input = ShogiLegalMoveInputLayer(embedding_dim=embedding_dim)
        self.policy_head = ShogiStateSummaryLegalMovePolicyHead(
            input_dim=embedding_dim * 2,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        *,
        position_hidden: torch.Tensor,
        legal_move_features: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> torch.Tensor:
        state_hidden = position_hidden[:, 0]
        move_embedding = self.move_input(legal_move_features)
        expanded_state = state_hidden[:, None, :].expand(-1, move_embedding.size(1), -1)
        return self.policy_head(torch.cat((expanded_state, move_embedding), dim=-1), legal_move_mask)


class ShogiLegalMoveAttentionPolicyHead(nn.Module):
    def __init__(self, *, embedding_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        *,
        position_hidden: torch.Tensor,
        legal_move_embeddings: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _weights = self.cross_attention(legal_move_embeddings, position_hidden, position_hidden, need_weights=False)
        logits = self.scorer(torch.cat((legal_move_embeddings, attended), dim=-1)).squeeze(-1)
        return logits.masked_fill(~legal_move_mask, torch.finfo(logits.dtype).min)


class ShogiLegalMoveAttentionPolicyOutput(nn.Module):
    def __init__(self, *, embedding_dim: int, num_heads: int, hidden_dim: int) -> None:
        super().__init__()
        self.move_input = ShogiLegalMoveInputLayer(embedding_dim=embedding_dim)
        self.policy_head = ShogiLegalMoveAttentionPolicyHead(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        *,
        position_hidden: torch.Tensor,
        legal_move_features: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> torch.Tensor:
        legal_move_embeddings = self.legal_move_embeddings(position_hidden, legal_move_features)
        return self.policy_head(
            position_hidden=position_hidden,
            legal_move_embeddings=legal_move_embeddings,
            legal_move_mask=legal_move_mask,
        )

    def legal_move_embeddings(self, position_hidden: torch.Tensor, legal_move_features: torch.Tensor) -> torch.Tensor:
        move_embedding = self.move_input(legal_move_features)
        from_square_hidden = _legal_move_square_hidden(
            position_hidden,
            legal_move_features[..., 0],
            zero_square_id=NO_FROM_SQUARE_ID,
        )
        to_square_hidden = _legal_move_square_hidden(position_hidden, legal_move_features[..., 1])
        return move_embedding + from_square_hidden + to_square_hidden


def _legal_move_square_hidden(
    position_hidden: torch.Tensor,
    square_ids: torch.Tensor,
    *,
    zero_square_id: int | None = None,
) -> torch.Tensor:
    embedding_dim = position_hidden.size(-1)
    zero_mask = square_ids.eq(zero_square_id) if zero_square_id is not None else torch.zeros_like(square_ids).bool()
    safe_square_ids = square_ids.masked_fill(zero_mask, 0)
    element_indices = safe_square_ids + SQUARE_ELEMENT_OFFSET
    gather_indices = element_indices[..., None].expand(-1, -1, embedding_dim)
    square_hidden = position_hidden.gather(dim=1, index=gather_indices)
    return square_hidden.masked_fill(zero_mask[..., None], 0.0)
