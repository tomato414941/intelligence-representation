from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from intrep.gpt_training import GPTTrainingConfig
from intrep.mixed_corpus import MixedDocument, default_mixed_documents


DocumentLoader = Callable[[str | Path], list[MixedDocument]]


@dataclass(frozen=True)
class CorpusSelection:
    label: str
    documents: list[MixedDocument]


def _load_file_documents(path: str | Path) -> list[MixedDocument]:
    try:
        from intrep.mixed_corpus import load_mixed_documents_jsonl
    except ImportError as error:
        raise RuntimeError("file-backed corpus loading is not available in this build") from error
    return load_mixed_documents_jsonl(path)


def _load_builtin_grid_documents() -> list[MixedDocument]:
    from intrep import grid_corpus

    return grid_corpus.default_grid_documents()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate next-observation ranking before and after tiny GPT training."
    )
    parser.add_argument(
        "--corpus",
        choices=("builtin", "builtin-grid", "file"),
        default="builtin",
        help="Use a built-in corpus or load documents from a JSONL file.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Path to a mixed-document JSONL corpus when --corpus=file.",
    )
    parser.add_argument(
        "--eval-corpus-path",
        type=Path,
        help="Optional mixed-document JSONL corpus for held-out ranking evaluation.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help="Write learning and ranking evaluation metrics to a JSON file.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    document_loader: DocumentLoader | None = None,
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    loader = document_loader or _load_file_documents
    corpus = _select_corpus(
        args.corpus,
        args.corpus_path,
        parser=parser,
        document_loader=loader,
    )
    eval_corpus = None
    if args.eval_corpus_path is not None:
        try:
            eval_documents = loader(args.eval_corpus_path)
        except RuntimeError as error:
            parser.error(str(error))
        except (OSError, ValueError) as error:
            parser.error(f"could not load eval corpus from {args.eval_corpus_path}: {error}")
        eval_corpus = CorpusSelection(label=str(args.eval_corpus_path), documents=eval_documents)

    training_config = GPTTrainingConfig(
        context_length=args.context_length,
        batch_stride=args.batch_stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    result = evaluate_next_observation_learning(
        corpus.documents,
        eval_documents=eval_corpus.documents if eval_corpus is not None else None,
        training_config=training_config,
    )

    eval_label = eval_corpus.label if eval_corpus is not None else "train"
    print("intrep next-observation evaluation")
    top1_delta = result.after_metrics.top1_accuracy - result.before_metrics.top1_accuracy
    margin_delta = result.after_metrics.mean_margin - result.before_metrics.mean_margin
    print(
        f"corpus={corpus.label}"
        f" eval_corpus={eval_label}"
        f" train_cases={len(result.train_cases)}"
        f" eval_cases={len(result.eval_cases)}"
        f" tokens={result.training_result.token_count}"
        f" steps={result.training_result.steps}"
        f" before_top1_accuracy={result.before_metrics.top1_accuracy:.4f}"
        f" after_top1_accuracy={result.after_metrics.top1_accuracy:.4f}"
        f" top1_delta={top1_delta:.4f}"
        f" before_margin={result.before_metrics.mean_margin:.4f}"
        f" after_margin={result.after_metrics.mean_margin:.4f}"
        f" margin_delta={margin_delta:.4f}"
    )
    if args.metrics_path is not None:
        _write_metrics(args.metrics_path, result, corpus.label, eval_label, training_config)


def _select_corpus(
    corpus: str,
    corpus_path: Path | None,
    *,
    parser: argparse.ArgumentParser,
    document_loader: DocumentLoader,
) -> CorpusSelection:
    if corpus == "builtin":
        if corpus_path is not None:
            parser.error("--corpus-path can only be used with --corpus=file")
        return CorpusSelection(label="builtin", documents=default_mixed_documents())

    if corpus == "builtin-grid":
        if corpus_path is not None:
            parser.error("--corpus-path can only be used with --corpus=file")
        return CorpusSelection(label="builtin-grid", documents=_load_builtin_grid_documents())

    if corpus == "file":
        if corpus_path is None:
            parser.error("--corpus-path is required when --corpus=file")
        try:
            documents = document_loader(corpus_path)
        except RuntimeError as error:
            parser.error(str(error))
        except (OSError, ValueError) as error:
            parser.error(f"could not load corpus from {corpus_path}: {error}")
        return CorpusSelection(label=str(corpus_path), documents=documents)

    raise AssertionError(f"unsupported corpus selection: {corpus}")


def evaluate_next_observation_learning(
    documents: list[MixedDocument],
    *,
    eval_documents: list[MixedDocument] | None = None,
    training_config: GPTTrainingConfig,
) -> object:
    from intrep.next_observation_evaluation import (
        evaluate_next_observation_learning as evaluate,
    )

    return evaluate(documents, eval_documents=eval_documents, training_config=training_config)


def _write_metrics(
    path: Path,
    result: object,
    corpus_label: str,
    eval_label: str,
    training_config: GPTTrainingConfig,
) -> None:
    before_metrics = getattr(result, "before_metrics")
    after_metrics = getattr(result, "after_metrics")
    payload = {
        "corpus": corpus_label,
        "eval_corpus": eval_label,
        "train_case_count": len(getattr(result, "train_cases")),
        "eval_case_count": len(getattr(result, "eval_cases")),
        "training": _training_result_to_dict(
            getattr(result, "training_result"),
            training_config,
        ),
        "ranking": {
            "before": _ranking_summary_to_dict(result, "before", before_metrics),
            "after": _ranking_summary_to_dict(result, "after", after_metrics),
        },
        "deltas": {
            "top1_accuracy": after_metrics.top1_accuracy - before_metrics.top1_accuracy,
            "mean_margin": after_metrics.mean_margin - before_metrics.mean_margin,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_result_to_dict(
    result: object,
    training_config: GPTTrainingConfig,
) -> dict[str, object]:
    return {
        "steps": getattr(result, "steps"),
        "token_count": getattr(result, "token_count"),
        "context_length": training_config.context_length,
        "batch_size": training_config.batch_size,
        "batch_stride": training_config.batch_stride,
        "learning_rate": training_config.learning_rate,
        "seed": training_config.seed,
        "initial_loss": getattr(result, "initial_loss", None),
        "final_loss": getattr(result, "final_loss", None),
        "best_loss": getattr(result, "best_loss", None),
        "loss_reduction": getattr(result, "loss_reduction", None),
        "loss_reduction_ratio": getattr(result, "loss_reduction_ratio", None),
        "initial_train_loss": getattr(result, "initial_train_loss", None),
        "final_train_loss": getattr(result, "final_train_loss", None),
        "initial_eval_loss": getattr(result, "initial_eval_loss", None),
        "final_eval_loss": getattr(result, "final_eval_loss", None),
    }


def _ranking_summary_to_dict(
    result: object,
    prefix: str,
    fallback_metrics: object,
) -> dict[str, object]:
    summary = getattr(result, f"{prefix}_summary", None)
    if summary is None:
        return {"overall": _ranking_metrics_to_dict(fallback_metrics)}
    return {
        "overall": _ranking_metrics_to_dict(getattr(summary, "overall")),
        "per_modality": {
            modality: _ranking_metrics_to_dict(metrics)
            for modality, metrics in getattr(summary, "per_modality").items()
        },
        "modality_counts": dict(getattr(summary, "modality_counts")),
    }


def _ranking_metrics_to_dict(metrics: object) -> dict[str, float]:
    return {
        "top1_accuracy": getattr(metrics, "top1_accuracy"),
        "mean_positive_loss": getattr(metrics, "mean_positive_loss"),
        "mean_best_distractor_loss": getattr(metrics, "mean_best_distractor_loss"),
        "mean_margin": getattr(metrics, "mean_margin"),
    }


if __name__ == "__main__":
    main()
