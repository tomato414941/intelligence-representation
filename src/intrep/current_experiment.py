from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intrep.corpus_coverage import coverage_to_dict, summarize_corpus_coverage
from intrep.gpt_model import GPTConfig
from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt
from intrep.language_modeling_metrics import language_modeling_metrics_from_training_result
from intrep.mixed_corpus import (
    MixedDocument,
    default_mixed_documents,
    load_mixed_documents_jsonl,
)
from intrep.next_observation_cases import extract_next_observation_cases
from intrep.next_observation_evaluation import evaluate_next_observation_learning


DocumentLoader = Callable[[str | Path], list[MixedDocument]]
EvaluationRunner = Callable[..., Any]


@dataclass(frozen=True)
class ExperimentCorpus:
    label: str
    documents: list[MixedDocument]


def select_corpus(
    corpus: str,
    corpus_path: Path | None = None,
    *,
    document_loader: DocumentLoader = load_mixed_documents_jsonl,
) -> ExperimentCorpus:
    if corpus == "builtin":
        if corpus_path is not None:
            raise ValueError("--corpus-path can only be used with --corpus=file")
        return ExperimentCorpus(label="builtin", documents=default_mixed_documents())

    if corpus == "builtin-grid":
        if corpus_path is not None:
            raise ValueError("--corpus-path can only be used with --corpus=file")
        from intrep.grid_corpus import default_grid_documents

        return ExperimentCorpus(label="builtin-grid", documents=default_grid_documents())

    if corpus == "file":
        if corpus_path is None:
            raise ValueError("--corpus-path is required when --corpus=file")
        return ExperimentCorpus(label=str(corpus_path), documents=document_loader(corpus_path))

    raise ValueError(f"unsupported corpus: {corpus}")


def run_current_experiment(
    documents: Sequence[MixedDocument],
    *,
    eval_documents: Sequence[MixedDocument] | None = None,
    corpus_label: str = "builtin",
    eval_corpus_label: str | None = None,
    training_config: GPTTrainingConfig | None = None,
    model_config: GPTConfig | None = None,
    evaluation_runner: EvaluationRunner = evaluate_next_observation_learning,
) -> dict[str, object]:
    config = training_config or GPTTrainingConfig()
    train_documents = list(documents)
    held_out_documents = list(eval_documents) if eval_documents is not None else None
    eval_label = (
        eval_corpus_label
        if eval_corpus_label is not None
        else ("eval" if held_out_documents is not None else "train")
    )
    eval_case_documents = held_out_documents if held_out_documents is not None else train_documents
    train_cases = extract_next_observation_cases(train_documents)
    eval_cases = extract_next_observation_cases(eval_case_documents)

    if len(eval_cases) >= 2:
        evaluation = evaluation_runner(
            train_documents,
            eval_documents=held_out_documents,
            training_config=config,
            model_config=model_config,
        )
        return _summary_from_evaluation(
            evaluation,
            train_documents=train_documents,
            eval_documents=held_out_documents,
            corpus_label=corpus_label,
            eval_label=eval_label,
            training_config=config,
        )

    training_result = train_mixed_gpt(
        documents=train_documents,
        eval_documents=held_out_documents,
        training_config=config,
        model_config=model_config,
    )
    return _summary_from_training_only(
        training_result,
        train_documents=train_documents,
        eval_documents=held_out_documents,
        corpus_label=corpus_label,
        eval_label=eval_label,
        training_config=config,
        train_case_count=len(train_cases),
        eval_case_count=len(eval_cases),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the current corpus learning and next-observation evaluation path."
    )
    parser.add_argument(
        "--corpus",
        choices=("builtin", "builtin-grid", "file"),
        default="builtin",
    )
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--eval-corpus-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    document_loader: DocumentLoader = load_mixed_documents_jsonl,
    evaluation_runner: EvaluationRunner = evaluate_next_observation_learning,
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        corpus = select_corpus(
            args.corpus,
            args.corpus_path,
            document_loader=document_loader,
        )
        eval_corpus = (
            ExperimentCorpus(
                label=str(args.eval_corpus_path),
                documents=document_loader(args.eval_corpus_path),
            )
            if args.eval_corpus_path is not None
            else None
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    summary = run_current_experiment(
        corpus.documents,
        eval_documents=eval_corpus.documents if eval_corpus is not None else None,
        corpus_label=corpus.label,
        eval_corpus_label=eval_corpus.label if eval_corpus is not None else None,
        training_config=GPTTrainingConfig(
            context_length=args.context_length,
            batch_stride=args.batch_stride,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        evaluation_runner=evaluation_runner,
    )

    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


def _summary_from_evaluation(
    evaluation: object,
    *,
    train_documents: Sequence[MixedDocument],
    eval_documents: Sequence[MixedDocument] | None,
    corpus_label: str,
    eval_label: str,
    training_config: GPTTrainingConfig,
) -> dict[str, object]:
    training_result = getattr(evaluation, "training_result")
    before_metrics = getattr(evaluation, "before_metrics")
    after_metrics = getattr(evaluation, "after_metrics")
    before_summary = getattr(evaluation, "before_summary", None)
    after_summary = getattr(evaluation, "after_summary", None)
    return {
        "corpus": _corpus_dict(corpus_label, eval_label),
        "coverage": _coverage_dict(train_documents, eval_documents),
        "training_config": _training_config_to_dict(training_config),
        "training_loss": _training_result_to_dict(training_result),
        "language_modeling": language_modeling_metrics_from_training_result(training_result),
        "next_observation": {
            "status": "evaluated",
            "train_case_count": len(getattr(evaluation, "train_cases")),
            "eval_case_count": len(getattr(evaluation, "eval_cases")),
            "before": _ranking_metrics_to_dict(before_metrics),
            "after": _ranking_metrics_to_dict(after_metrics),
            "delta": {
                "top1_accuracy": after_metrics.top1_accuracy - before_metrics.top1_accuracy,
                "mean_margin": after_metrics.mean_margin - before_metrics.mean_margin,
            },
            "modality_counts": _modality_counts(before_summary, after_summary),
        },
    }


def _summary_from_training_only(
    training_result: object,
    *,
    train_documents: Sequence[MixedDocument],
    eval_documents: Sequence[MixedDocument] | None,
    corpus_label: str,
    eval_label: str,
    training_config: GPTTrainingConfig,
    train_case_count: int,
    eval_case_count: int,
) -> dict[str, object]:
    return {
        "corpus": _corpus_dict(corpus_label, eval_label),
        "coverage": _coverage_dict(train_documents, eval_documents),
        "training_config": _training_config_to_dict(training_config),
        "training_loss": _training_result_to_dict(training_result),
        "language_modeling": language_modeling_metrics_from_training_result(training_result),
        "next_observation": {
            "status": "skipped",
            "reason": "at least two next-observation cases are required",
            "train_case_count": train_case_count,
            "eval_case_count": eval_case_count,
        },
    }


def _corpus_dict(label: str, eval_label: str) -> dict[str, str]:
    return {"label": label, "eval_label": eval_label}


def _coverage_dict(
    train_documents: Sequence[MixedDocument],
    eval_documents: Sequence[MixedDocument] | None,
) -> dict[str, object]:
    payload = {"train": coverage_to_dict(summarize_corpus_coverage(train_documents))}
    if eval_documents is not None:
        payload["eval"] = coverage_to_dict(summarize_corpus_coverage(eval_documents))
    return payload


def _training_config_to_dict(config: GPTTrainingConfig) -> dict[str, object]:
    return {
        "context_length": config.context_length,
        "batch_size": config.batch_size,
        "batch_stride": config.batch_stride,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "seed": config.seed,
    }


def _training_result_to_dict(result: object) -> dict[str, object]:
    return {
        "initial_loss": getattr(result, "initial_loss", None),
        "final_loss": getattr(result, "final_loss", None),
        "steps": getattr(result, "steps"),
        "token_count": getattr(result, "token_count"),
        "best_loss": getattr(result, "best_loss", None),
        "loss_reduction": getattr(result, "loss_reduction", None),
        "loss_reduction_ratio": getattr(result, "loss_reduction_ratio", None),
        "loss_history": list(getattr(result, "loss_history", ())),
        "initial_train_loss": getattr(result, "initial_train_loss", None),
        "final_train_loss": getattr(result, "final_train_loss", None),
        "initial_eval_loss": getattr(result, "initial_eval_loss", None),
        "final_eval_loss": getattr(result, "final_eval_loss", None),
    }


def _ranking_metrics_to_dict(metrics: object) -> dict[str, float]:
    return {
        "top1_accuracy": getattr(metrics, "top1_accuracy"),
        "mean_positive_loss": getattr(metrics, "mean_positive_loss"),
        "mean_best_distractor_loss": getattr(metrics, "mean_best_distractor_loss"),
        "mean_margin": getattr(metrics, "mean_margin"),
    }


def _modality_counts(before_summary: object | None, after_summary: object | None) -> dict[str, int]:
    summary = after_summary if after_summary is not None else before_summary
    if summary is None:
        return {}
    return dict(getattr(summary, "modality_counts", {}))


if __name__ == "__main__":
    main()
