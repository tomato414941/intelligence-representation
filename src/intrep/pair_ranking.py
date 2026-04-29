from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from intrep.byte_tokenizer import ByteTokenizer
from intrep.token_scoring import next_token_losses


ContinuationScorer = Callable[[Any, ByteTokenizer, str, str], float]


def torch_next_token_continuation_loss(
    model: Any,
    tokenizer: ByteTokenizer,
    prefix: str,
    continuation: str,
) -> float:
    return torch_next_token_continuation_losses(model, tokenizer, prefix, [continuation])[0]


def torch_next_token_continuation_losses(
    model: Any,
    tokenizer: ByteTokenizer,
    prefix: str,
    continuations: Sequence[str],
) -> list[float]:
    import torch

    prefix_ids = tokenizer.encode(prefix)
    if not prefix_ids:
        raise ValueError("prefix must encode to at least one token")
    continuation_id_rows = [tokenizer.encode(continuation) for continuation in continuations]
    if not continuation_id_rows:
        raise ValueError("continuations must not be empty")
    if any(not continuation_ids for continuation_ids in continuation_id_rows):
        raise ValueError("continuation must encode to at least one token")

    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    parameter = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
    device = parameter.device if parameter is not None else torch.device("cpu")
    max_token_count = max(len(prefix_ids) + len(continuation_ids) for continuation_ids in continuation_id_rows)
    context_length = int(getattr(getattr(model, "config", None), "context_length", max_token_count))

    try:
        with torch.no_grad():
            windows_by_length: dict[int, list[tuple[int, list[int]]]] = {}
            targets_by_length: dict[int, list[int]] = {}
            for continuation_index, continuation_ids in enumerate(continuation_id_rows):
                token_ids = prefix_ids + continuation_ids
                if len(token_ids) < 2:
                    raise ValueError("prefix and continuation must encode to at least two tokens")
                for target_index in range(len(prefix_ids), len(token_ids)):
                    window_start = max(0, target_index - context_length)
                    window = token_ids[window_start:target_index]
                    windows_by_length.setdefault(len(window), []).append((continuation_index, window))
                    targets_by_length.setdefault(len(window), []).append(token_ids[target_index])

            total_losses = [0.0 for _ in continuation_id_rows]
            total_counts = [0 for _ in continuation_id_rows]
            for length, windows in windows_by_length.items():
                input_ids = torch.tensor([window for _index, window in windows], dtype=torch.long, device=device)
                targets = torch.tensor(targets_by_length[length], dtype=torch.long, device=device)
                scoring_token_ids = torch.cat([input_ids, targets.unsqueeze(1)], dim=1)
                model_logits = model(input_ids)
                scoring_logits = torch.cat(
                    [
                        model_logits,
                        torch.zeros(
                            (input_ids.size(0), 1, model_logits.size(2)),
                            dtype=model_logits.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
                loss_mask = torch.zeros_like(scoring_token_ids, dtype=torch.bool)
                loss_mask[:, -1] = True
                losses = next_token_losses(
                    scoring_logits,
                    scoring_token_ids,
                    loss_mask=loss_mask,
                )
                for (continuation_index, _window), loss in zip(windows, losses, strict=True):
                    total_losses[continuation_index] += float(loss.item())
                    total_counts[continuation_index] += 1
            return [
                total_loss / total_count
                for total_loss, total_count in zip(total_losses, total_counts, strict=True)
            ]
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
