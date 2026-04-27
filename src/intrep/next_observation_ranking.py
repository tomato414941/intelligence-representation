from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intrep.byte_tokenizer import ByteTokenizer
from intrep.next_observation_cases import NextObservationCase
from intrep.pair_ranking import ContinuationScorer


@dataclass(frozen=True)
class NextObservationRankingMetrics:
    top1_accuracy: float
    mean_positive_loss: float
    mean_best_distractor_loss: float
    mean_margin: float


def evaluate_next_observation_ranking(
    cases: Sequence[NextObservationCase],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    score_continuation_loss: ContinuationScorer | None = None,
) -> NextObservationRankingMetrics:
    if not cases:
        raise ValueError("cases must not be empty")
    if len(cases) < 2:
        raise ValueError("at least two cases are required to build distractors")

    scorer = score_continuation_loss or torch_context_limited_continuation_loss
    positive_losses: list[float] = []
    best_distractor_losses: list[float] = []
    margins: list[float] = []
    top1_count = 0

    for index, case in enumerate(cases):
        positive_loss = scorer(model, tokenizer, case.prefix, case.positive_next)
        distractor_losses = [
            scorer(model, tokenizer, case.prefix, distractor.positive_next)
            for distractor_index, distractor in enumerate(cases)
            if distractor_index != index
        ]
        if not distractor_losses:
            raise ValueError("each case requires at least one distractor")

        best_distractor_loss = min(distractor_losses)
        margin = best_distractor_loss - positive_loss

        positive_losses.append(positive_loss)
        best_distractor_losses.append(best_distractor_loss)
        margins.append(margin)
        if positive_loss < best_distractor_loss:
            top1_count += 1

    count = len(cases)
    return NextObservationRankingMetrics(
        top1_accuracy=top1_count / count,
        mean_positive_loss=sum(positive_losses) / count,
        mean_best_distractor_loss=sum(best_distractor_losses) / count,
        mean_margin=sum(margins) / count,
    )


def torch_context_limited_continuation_loss(
    model: Any,
    tokenizer: ByteTokenizer,
    prefix: str,
    continuation: str,
) -> float:
    import torch
    import torch.nn.functional as F

    prefix_ids = tokenizer.encode(prefix)
    continuation_ids = tokenizer.encode(continuation)
    if not prefix_ids:
        raise ValueError("prefix must encode to at least one token")
    if not continuation_ids:
        raise ValueError("continuation must encode to at least one token")

    context_length = getattr(getattr(model, "config", None), "context_length", None)
    if context_length is None:
        context_length = len(prefix_ids) + len(continuation_ids)
    if context_length <= 0:
        raise ValueError("model context length must be positive")

    token_ids = prefix_ids + continuation_ids
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()

    parameter = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
    device = parameter.device if parameter is not None else torch.device("cpu")

    try:
        total_loss = 0.0
        with torch.no_grad():
            for target_index in range(len(prefix_ids), len(token_ids)):
                start = max(0, target_index - context_length)
                input_ids = token_ids[start:target_index]
                if not input_ids:
                    raise ValueError("prefix and continuation must encode to at least two tokens")
                inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
                target = torch.tensor([token_ids[target_index]], dtype=torch.long, device=device)
                logits = model(inputs)[0, -1].unsqueeze(0)
                total_loss += float(F.cross_entropy(logits, target).item())
        return total_loss / len(continuation_ids)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
