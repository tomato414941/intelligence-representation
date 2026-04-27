from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from intrep.byte_tokenizer import ByteTokenizer
from intrep.next_observation_cases import NextObservationCase
from intrep.pair_ranking import ContinuationScorer

DISTRACTOR_POLICIES = ("all_other", "hard", "same_entity")


@dataclass(frozen=True)
class NextObservationRankingMetrics:
    top1_accuracy: float
    mean_positive_loss: float
    mean_best_distractor_loss: float
    mean_margin: float


@dataclass(frozen=True)
class NextObservationRankingSummary:
    overall: NextObservationRankingMetrics
    per_modality: dict[str, NextObservationRankingMetrics]
    modality_counts: dict[str, int]
    fallback_counts: dict[str, int]


@dataclass(frozen=True)
class _ScoredNextObservationCase:
    modality: str
    positive_loss: float
    best_distractor_loss: float
    margin: float
    top1_correct: bool
    fallback_key: str | None = None


def evaluate_next_observation_ranking(
    cases: Sequence[NextObservationCase],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    distractor_policy: str = "hard",
    score_continuation_loss: ContinuationScorer | None = None,
) -> NextObservationRankingMetrics:
    scored_cases = _score_next_observation_cases(
        cases,
        model,
        tokenizer,
        distractor_policy=distractor_policy,
        score_continuation_loss=score_continuation_loss,
    )
    return _summarize_scored_cases(scored_cases)


def evaluate_next_observation_ranking_summary(
    cases: Sequence[NextObservationCase],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    distractor_policy: str = "hard",
    score_continuation_loss: ContinuationScorer | None = None,
) -> NextObservationRankingSummary:
    scored_cases = _score_next_observation_cases(
        cases,
        model,
        tokenizer,
        distractor_policy=distractor_policy,
        score_continuation_loss=score_continuation_loss,
    )

    per_modality_cases: dict[str, list[_ScoredNextObservationCase]] = {}
    for scored_case in scored_cases:
        per_modality_cases.setdefault(scored_case.modality, []).append(scored_case)

    return NextObservationRankingSummary(
        overall=_summarize_scored_cases(scored_cases),
        per_modality={
            modality: _summarize_scored_cases(modality_cases)
            for modality, modality_cases in per_modality_cases.items()
        },
        modality_counts={
            modality: len(modality_cases)
            for modality, modality_cases in per_modality_cases.items()
        },
        fallback_counts=_summarize_fallback_counts(scored_cases),
    )


def _score_next_observation_cases(
    cases: Sequence[NextObservationCase],
    model: Any,
    tokenizer: ByteTokenizer,
    *,
    distractor_policy: str = "hard",
    score_continuation_loss: ContinuationScorer | None = None,
) -> list[_ScoredNextObservationCase]:
    if not cases:
        raise ValueError("cases must not be empty")
    if len(cases) < 2:
        raise ValueError("at least two cases are required to build distractors")
    if distractor_policy not in DISTRACTOR_POLICIES:
        raise ValueError(f"distractor_policy must be one of: {', '.join(DISTRACTOR_POLICIES)}")

    scorer = score_continuation_loss or torch_context_limited_continuation_loss
    scored_cases: list[_ScoredNextObservationCase] = []

    for index, case in enumerate(cases):
        positive_loss = scorer(model, tokenizer, case.prefix, case.positive_next)
        distractors, fallback_key = _select_distractors(cases, index, distractor_policy)
        distractor_losses = [
            scorer(model, tokenizer, case.prefix, distractor)
            for distractor in distractors
        ]
        if not distractor_losses:
            raise ValueError("each case requires at least one distractor")

        best_distractor_loss = min(distractor_losses)
        margin = best_distractor_loss - positive_loss

        scored_cases.append(
            _ScoredNextObservationCase(
                modality=case.modality,
                positive_loss=positive_loss,
                best_distractor_loss=best_distractor_loss,
                margin=margin,
                top1_correct=positive_loss < best_distractor_loss,
                fallback_key=fallback_key,
            )
        )

    return scored_cases


def _select_distractors(
    cases: Sequence[NextObservationCase],
    case_index: int,
    distractor_policy: str,
) -> tuple[list[str], str | None]:
    case = cases[case_index]
    all_other = [
        distractor
        for distractor_index, distractor in enumerate(cases)
        if distractor_index != case_index
    ]
    all_other_nexts = [distractor.positive_next for distractor in all_other]
    if distractor_policy == "all_other":
        return [*case.hard_negative_nexts, *all_other_nexts], None

    if distractor_policy == "same_entity":
        if case.hard_negative_nexts:
            return list(case.hard_negative_nexts), None
        same_entity_distractors = [
            distractor.positive_next
            for distractor in all_other
            if case.group_id is not None and distractor.group_id == case.group_id
        ]
        if same_entity_distractors:
            return same_entity_distractors, None
        hard_distractors, fallback_key = _select_hard_distractors(
            case,
            all_other,
            all_other_nexts,
        )
        return hard_distractors, fallback_key or "same_entity_to_hard"

    return _select_hard_distractors(case, all_other, all_other_nexts)


def _select_hard_distractors(
    case: NextObservationCase,
    all_other: Sequence[NextObservationCase],
    all_other_nexts: Sequence[str],
) -> tuple[list[str], str | None]:
    same_modality_distractors = [
        distractor.positive_next
        for distractor in all_other
        if distractor.modality == case.modality
    ]
    if same_modality_distractors:
        return [*case.hard_negative_nexts, *same_modality_distractors], None
    return [*case.hard_negative_nexts, *all_other_nexts], "hard_to_all_other"


def _summarize_scored_cases(
    scored_cases: Sequence[_ScoredNextObservationCase],
) -> NextObservationRankingMetrics:
    if not scored_cases:
        raise ValueError("scored_cases must not be empty")

    count = len(scored_cases)
    return NextObservationRankingMetrics(
        top1_accuracy=sum(scored_case.top1_correct for scored_case in scored_cases)
        / count,
        mean_positive_loss=sum(
            scored_case.positive_loss for scored_case in scored_cases
        )
        / count,
        mean_best_distractor_loss=sum(
            scored_case.best_distractor_loss for scored_case in scored_cases
        )
        / count,
        mean_margin=sum(scored_case.margin for scored_case in scored_cases) / count,
    )


def _summarize_fallback_counts(
    scored_cases: Sequence[_ScoredNextObservationCase],
) -> dict[str, int]:
    fallback_counts: dict[str, int] = {}
    for scored_case in scored_cases:
        if scored_case.fallback_key is None:
            continue
        fallback_counts[scored_case.fallback_key] = (
            fallback_counts.get(scored_case.fallback_key, 0) + 1
        )
    return fallback_counts


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
