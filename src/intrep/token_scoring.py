from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Average next-token loss over masked target positions."""
    return next_token_losses(logits, target_token_ids, loss_mask=loss_mask).mean()


def next_token_losses(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row next-token loss over masked target positions."""
    _validate_next_token_loss_inputs(logits, target_token_ids, loss_mask)
    prediction_logits = logits[:, :-1, :]
    prediction_targets = target_token_ids[:, 1:]
    prediction_mask = loss_mask[:, 1:]
    mask_counts = prediction_mask.sum(dim=1)
    if bool((mask_counts == 0).any().item()):
        raise ValueError("loss_mask must include at least one predictable token per row")

    losses = F.cross_entropy(
        prediction_logits.reshape(-1, prediction_logits.size(-1)),
        prediction_targets.reshape(-1),
        reduction="none",
    ).reshape_as(prediction_targets)
    masked_losses = losses * prediction_mask.to(losses.dtype)
    return masked_losses.sum(dim=1) / mask_counts.to(losses.dtype)


def _validate_next_token_loss_inputs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if target_token_ids.ndim != 2:
        raise ValueError("target_token_ids must have shape [batch, sequence]")
    if loss_mask.ndim != 2:
        raise ValueError("loss_mask must have shape [batch, sequence]")
    if target_token_ids.dtype != torch.long:
        raise ValueError("target_token_ids must have dtype torch.long")
    if loss_mask.dtype != torch.bool:
        raise ValueError("loss_mask must have dtype torch.bool")
    if logits.size(0) != target_token_ids.size(0) or logits.size(1) != target_token_ids.size(1):
        raise ValueError("logits and target_token_ids must share batch and sequence size")
    if loss_mask.shape != target_token_ids.shape:
        raise ValueError("loss_mask must match target_token_ids shape")
    if logits.size(1) < 2:
        raise ValueError("sequence length must be at least 2")
    if logits.size(2) <= 0:
        raise ValueError("vocab size must be positive")
    min_id = int(target_token_ids.min().item())
    max_id = int(target_token_ids.max().item())
    if min_id < 0 or max_id >= logits.size(2):
        raise ValueError("target_token_ids values must be in the logits vocabulary range")
