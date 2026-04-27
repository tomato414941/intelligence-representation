from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import DecoderOnlyGPT, GPTConfig
from intrep.gpt_training import (
    GPTTrainingConfig,
    GPTTrainingResult,
    train_mixed_gpt_with_artifacts,
)
from intrep.mixed_corpus import MixedDocument
from intrep.next_observation_cases import (
    NextObservationCase,
    extract_next_observation_cases,
)
from intrep.next_observation_ranking import (
    NextObservationRankingMetrics,
    NextObservationRankingSummary,
    evaluate_next_observation_ranking_summary,
)
from intrep.pair_ranking import ContinuationScorer


@dataclass(frozen=True)
class NextObservationEvaluationResult:
    train_cases: list[NextObservationCase]
    eval_cases: list[NextObservationCase]
    before_metrics: NextObservationRankingMetrics
    after_metrics: NextObservationRankingMetrics
    before_summary: NextObservationRankingSummary
    after_summary: NextObservationRankingSummary
    training_result: GPTTrainingResult

    @property
    def case_count(self) -> int:
        return len(self.eval_cases)


def evaluate_next_observation_learning(
    documents: Sequence[MixedDocument],
    eval_documents: Sequence[MixedDocument] | None = None,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    *,
    score_continuation_loss: ContinuationScorer | None = None,
) -> NextObservationEvaluationResult:
    config = training_config or GPTTrainingConfig()
    corpus_documents = list(documents)
    evaluation_documents = list(eval_documents) if eval_documents is not None else corpus_documents
    train_cases = extract_next_observation_cases(corpus_documents)
    eval_cases = extract_next_observation_cases(evaluation_documents)
    tokenizer = ByteTokenizer()
    before_model = _build_untrained_model(
        tokenizer=tokenizer,
        training_config=config,
        model_config=model_config,
    )
    before_summary = evaluate_next_observation_ranking_summary(
        eval_cases,
        model=before_model,
        tokenizer=tokenizer,
        score_continuation_loss=score_continuation_loss,
    )
    artifacts = train_mixed_gpt_with_artifacts(
        documents=corpus_documents,
        eval_documents=list(eval_documents) if eval_documents is not None else None,
        training_config=config,
        model_config=model_config,
    )
    after_summary = evaluate_next_observation_ranking_summary(
        eval_cases,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        score_continuation_loss=score_continuation_loss,
    )

    return NextObservationEvaluationResult(
        train_cases=train_cases,
        eval_cases=eval_cases,
        before_metrics=before_summary.overall,
        after_metrics=after_summary.overall,
        before_summary=before_summary,
        after_summary=after_summary,
        training_result=artifacts.result,
    )


def _build_untrained_model(
    *,
    tokenizer: ByteTokenizer,
    training_config: GPTTrainingConfig,
    model_config: GPTConfig | None,
) -> DecoderOnlyGPT:
    torch.manual_seed(training_config.seed)
    config = model_config or GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=training_config.context_length,
    )
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model_config.vocab_size must match the byte tokenizer vocab size")
    if config.context_length != training_config.context_length:
        raise ValueError("model_config.context_length must match training_config.context_length")
    return DecoderOnlyGPT(config)
