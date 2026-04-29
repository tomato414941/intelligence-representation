from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenSequence:
    token_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative integers")
        if self.loss_mask is not None and len(self.loss_mask) != len(self.token_ids):
            raise ValueError("loss_mask must match token_ids length")
        if self.loss_mask is not None and not any(self.loss_mask):
            raise ValueError("loss_mask must include at least one training token")


def token_sequence_from_ids(
    token_ids: list[int] | tuple[int, ...],
    *,
    loss_mask: list[bool] | tuple[bool, ...] | None = None,
) -> TokenSequence:
    return TokenSequence(
        token_ids=tuple(token_ids),
        loss_mask=tuple(loss_mask) if loss_mask is not None else None,
    )


@dataclass(frozen=True)
class HiddenSequence:
    embeddings: torch.Tensor
    loss_mask: tuple[bool, ...] | None = None

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape [sequence, hidden]")
        if self.embeddings.size(0) == 0:
            raise ValueError("embeddings sequence length must be positive")
        if self.embeddings.size(1) == 0:
            raise ValueError("embeddings hidden size must be positive")
        if not torch.is_floating_point(self.embeddings):
            raise ValueError("embeddings must be floating point")
        if self.loss_mask is not None and len(self.loss_mask) != self.embeddings.size(0):
            raise ValueError("loss_mask must match embeddings sequence length")
        if self.loss_mask is not None and not any(self.loss_mask):
            raise ValueError("loss_mask must include at least one training position")


def hidden_sequence_from_embeddings(
    embeddings: torch.Tensor,
    *,
    loss_mask: list[bool] | tuple[bool, ...] | None = None,
) -> HiddenSequence:
    return HiddenSequence(
        embeddings=embeddings,
        loss_mask=tuple(loss_mask) if loss_mask is not None else None,
    )
