from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intrep.byte_tokenizer import ByteTokenizer
from intrep.future_prediction_cases import (
    FuturePredictionCase,
    FuturePredictionRendering,
    render_future_prediction_texts,
)
from intrep.pair_ranking import ContinuationScorer, torch_next_token_continuation_loss
from intrep.pair_ranking import torch_next_token_continuation_losses


@dataclass(frozen=True)
class FuturePredictionRankingMetrics:
    top1_accuracy: float
    mean_positive_loss: float
    mean_best_negative_loss: float
    mean_margin: float


@dataclass(frozen=True)
class FuturePredictionRankingSummary:
    overall: FuturePredictionRankingMetrics
    by_condition: dict[str, FuturePredictionRankingMetrics]
    condition_counts: dict[str, int]


def evaluate_future_prediction_ranking(
    cases: Sequence[FuturePredictionCase],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    score_continuation_loss: ContinuationScorer | None = None,
    rendering: FuturePredictionRendering = "signal",
    image_patch_size: int = 1,
    image_channel_bins: int = 4,
    image_token_format: str = "flat",
    max_negatives: int | None = None,
) -> FuturePredictionRankingSummary:
    if not cases:
        raise ValueError("cases must not be empty")
    scorer = score_continuation_loss or torch_next_token_continuation_loss
    case_metrics = [
        _score_case(
            case,
            model=model,
            tokenizer=tokenizer,
            scorer=scorer,
            rendering=rendering,
            image_patch_size=image_patch_size,
            image_channel_bins=image_channel_bins,
            image_token_format=image_token_format,
            max_negatives=max_negatives,
        )
        for case in cases
    ]
    by_condition_losses: dict[str, list[tuple[float, float, float, bool]]] = defaultdict(list)
    for case, metrics in zip(cases, case_metrics):
        by_condition_losses[case.condition].append(metrics)
    return FuturePredictionRankingSummary(
        overall=_aggregate(case_metrics),
        by_condition={
            condition: _aggregate(metrics)
            for condition, metrics in by_condition_losses.items()
        },
        condition_counts={
            condition: len(metrics)
            for condition, metrics in by_condition_losses.items()
        },
    )


def _score_case(
    case: FuturePredictionCase,
    *,
    model: Any,
    tokenizer: ByteTokenizer,
    scorer: ContinuationScorer,
    rendering: FuturePredictionRendering,
    image_patch_size: int,
    image_channel_bins: int,
    image_token_format: str,
    max_negatives: int | None,
) -> tuple[float, float, float, bool]:
    if not case.negative_events:
        raise ValueError(f"case {case.id} must have at least one negative event")
    prefix, positive, negatives = render_future_prediction_texts(
        case,
        rendering=rendering,
        image_patch_size=image_patch_size,
        image_channel_bins=image_channel_bins,
        image_token_format=image_token_format,
    )
    if max_negatives is not None:
        if max_negatives <= 0:
            raise ValueError("max_negatives must be positive")
        negatives = negatives[:max_negatives]
    distinct_negatives = [negative for negative in negatives if negative != positive]
    if scorer is torch_next_token_continuation_loss:
        losses = torch_next_token_continuation_losses(
            model,
            tokenizer,
            prefix,
            [positive, *distinct_negatives],
        )
        positive_loss = losses[0]
        negative_losses = losses[1:]
    else:
        positive_loss = scorer(model, tokenizer, prefix, positive)
        negative_losses = [
            scorer(model, tokenizer, prefix, negative)
            for negative in distinct_negatives
        ]
    if not negative_losses:
        raise ValueError(f"case {case.id} must have at least one distinct negative event")
    best_negative_loss = min(negative_losses)
    margin = best_negative_loss - positive_loss
    return positive_loss, best_negative_loss, margin, positive_loss < best_negative_loss


def _aggregate(metrics: Sequence[tuple[float, float, float, bool]]) -> FuturePredictionRankingMetrics:
    count = len(metrics)
    return FuturePredictionRankingMetrics(
        top1_accuracy=sum(1 for *_, is_top1 in metrics if is_top1) / count,
        mean_positive_loss=sum(metric[0] for metric in metrics) / count,
        mean_best_negative_loss=sum(metric[1] for metric in metrics) / count,
        mean_margin=sum(metric[2] for metric in metrics) / count,
    )
