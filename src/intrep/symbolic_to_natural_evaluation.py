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
from intrep.mixed_corpus_evaluation import (
    MixedEnvironmentDocumentPair,
    extract_environment_document_pairs,
)
from intrep.pair_ranking import (
    ContinuationScorer,
    PairRankingMetrics,
    evaluate_symbolic_to_natural_ranking,
)


@dataclass(frozen=True)
class SymbolicToNaturalEvaluationResult:
    train_pairs: list[MixedEnvironmentDocumentPair]
    eval_pairs: list[MixedEnvironmentDocumentPair]
    before_metrics: PairRankingMetrics
    after_metrics: PairRankingMetrics
    training_result: GPTTrainingResult

    @property
    def pair_count(self) -> int:
        return len(self.eval_pairs)


def evaluate_symbolic_to_natural_learning(
    documents: Sequence[MixedDocument],
    eval_documents: Sequence[MixedDocument] | None = None,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    *,
    score_continuation_loss: ContinuationScorer | None = None,
) -> SymbolicToNaturalEvaluationResult:
    config = training_config or GPTTrainingConfig()
    corpus_documents = list(documents)
    evaluation_documents = list(eval_documents) if eval_documents is not None else corpus_documents
    train_pairs = extract_environment_document_pairs(corpus_documents)
    eval_pairs = extract_environment_document_pairs(evaluation_documents)
    if len(eval_pairs) < 2:
        raise ValueError("at least two eval environment pairs are required")

    tokenizer = ByteTokenizer()
    before_model = _build_untrained_model(
        tokenizer=tokenizer,
        training_config=config,
        model_config=model_config,
    )
    before_metrics = evaluate_symbolic_to_natural_ranking(
        eval_pairs,
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
    after_metrics = evaluate_symbolic_to_natural_ranking(
        eval_pairs,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        score_continuation_loss=score_continuation_loss,
    )

    return SymbolicToNaturalEvaluationResult(
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
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
