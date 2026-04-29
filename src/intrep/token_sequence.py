from __future__ import annotations

from dataclasses import dataclass


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
