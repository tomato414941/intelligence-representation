from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from intrep.gpt_training import GPTTrainingConfig
from intrep.mixed_corpus import MixedDocument, default_mixed_documents
from intrep.next_observation_ranking import DISTRACTOR_POLICIES


DocumentLoader = Callable[[str | Path], list[MixedDocument]]
CLI_DISTRACTOR_POLICIES = tuple(dict.fromkeys((*DISTRACTOR_POLICIES, "same_entity")))


@dataclass(frozen=True)
class CorpusSelection:
    label: str
    documents: list[MixedDocument]
    eval_label: str | None = None
    eval_documents: list[MixedDocument] | None = None


def _load_file_documents(path: str | Path) -> list[MixedDocument]:
    try:
        from intrep.signal_corpus import load_corpus_jsonl_as_mixed_documents
    except ImportError as error:
        raise RuntimeError("file-backed corpus loading is not available in this build") from error
    return load_corpus_jsonl_as_mixed_documents(path)


def _load_builtin_grid_documents() -> list[MixedDocument]:
    from intrep import grid_corpus

    return grid_corpus.default_grid_documents()


def _load_generated_environment_documents(
    eval_slice: str,
) -> tuple[list[MixedDocument], list[MixedDocument]]:
    from intrep.generated_environment_corpus import (
        generated_environment_corpus_selection,
    )

    selection = generated_environment_corpus_selection(eval_slice)
    return selection.train_documents, selection.eval_documents


def build_parser() -> argparse.ArgumentParser:
    from intrep.generated_environment_corpus import EVAL_SLICES

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate next-observation ranking as a world-modeling target "
            "inside the predictive token machine scaffold."
        )
    )
    parser.add_argument(
        "--corpus",
        choices=("builtin", "builtin-grid", "generated-environment", "file"),
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
    parser.add_argument(
        "--corpus-format",
        choices=("auto", "mixed-document", "signal", "typed-event"),
        default="auto",
        help="JSONL schema for --corpus=file. auto detects signal records by channel or role.",
    )
    parser.add_argument(
        "--render-format",
        choices=("plain", "signal-tags", "typed-tags", "image-tokens"),
        default="plain",
        help="Render corpus records as legacy plain documents, signal tags, or image token text.",
    )
    parser.add_argument(
        "--generated-eval-slice",
        choices=EVAL_SLICES,
        default="generated_held_out_object",
        help="Generated-environment eval slice when --corpus=generated-environment.",
    )
    parser.add_argument(
        "--distractor-policy",
        choices=CLI_DISTRACTOR_POLICIES,
        default="hard",
        help="Use all other continuations or only same-modality hard distractors.",
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


def _load_documents(
    path: str | Path,
    *,
    loader: DocumentLoader,
    custom_loader: bool,
    corpus_format: str,
    render_format: str,
) -> list[MixedDocument]:
    if not custom_loader:
        from intrep.signal_corpus import load_corpus_jsonl_as_mixed_documents

        return load_corpus_jsonl_as_mixed_documents(
            path,
            corpus_format=corpus_format,
            render_format=render_format,
        )
    documents = loader(path)
    if render_format == "plain":
        return documents
    if render_format not in ("signal-tags", "typed-tags", "image-tokens"):
        raise ValueError("render_format must be plain, signal-tags, typed-tags, or image-tokens")
    from intrep.signal_corpus import mixed_documents_to_signals, signals_to_mixed_documents

    return signals_to_mixed_documents(mixed_documents_to_signals(documents))


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
        args.generated_eval_slice,
        parser=parser,
        document_loader=loader,
        custom_loader=document_loader is not None,
        corpus_format=args.corpus_format,
        render_format=args.render_format,
    )
    eval_corpus = (
        CorpusSelection(label=corpus.eval_label, documents=corpus.eval_documents)
        if corpus.eval_label is not None and corpus.eval_documents is not None
        else None
    )
    if args.eval_corpus_path is not None:
        if eval_corpus is not None:
            parser.error("--eval-corpus-path cannot be used with --corpus=generated-environment")
        try:
            eval_documents = _load_documents(
                args.eval_corpus_path,
                loader=loader,
                custom_loader=document_loader is not None,
                corpus_format=args.corpus_format,
                render_format=args.render_format,
            )
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
        distractor_policy=args.distractor_policy,
    )

    eval_label = eval_corpus.label if eval_corpus is not None else "train"
    eval_split, generalization_eval = _eval_reporting(eval_label)
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
        f" eval_split={eval_split}"
        f" generalization_eval={str(generalization_eval).lower()}"
        f" distractor_policy={args.distractor_policy}"
        f" corpus_format={args.corpus_format}"
        f" render_format={args.render_format}"
        f" before_top1_accuracy={result.before_metrics.top1_accuracy:.4f}"
        f" after_top1_accuracy={result.after_metrics.top1_accuracy:.4f}"
        f" top1_delta={top1_delta:.4f}"
        f" before_margin={result.before_metrics.mean_margin:.4f}"
        f" after_margin={result.after_metrics.mean_margin:.4f}"
        f" margin_delta={margin_delta:.4f}"
    )
    if args.metrics_path is not None:
        _write_metrics(
            args.metrics_path,
            result,
            corpus.label,
            eval_label,
            args.distractor_policy,
            training_config,
        )


def _select_corpus(
    corpus: str,
    corpus_path: Path | None,
    generated_eval_slice: str,
    *,
    parser: argparse.ArgumentParser,
    document_loader: DocumentLoader,
    custom_loader: bool,
    corpus_format: str,
    render_format: str,
) -> CorpusSelection:
    if corpus == "builtin":
        if corpus_path is not None:
            parser.error("--corpus-path can only be used with --corpus=file")
        return CorpusSelection(label="builtin", documents=default_mixed_documents())

    if corpus == "builtin-grid":
        if corpus_path is not None:
            parser.error("--corpus-path can only be used with --corpus=file")
        return CorpusSelection(label="builtin-grid", documents=_load_builtin_grid_documents())

    if corpus == "generated-environment":
        if corpus_path is not None:
            parser.error("--corpus-path can only be used with --corpus=file")
        try:
            train_documents, eval_documents = _load_generated_environment_documents(
                generated_eval_slice
            )
        except ValueError as error:
            parser.error(str(error))
        return CorpusSelection(
            label="generated-environment",
            documents=train_documents,
            eval_label=generated_eval_slice,
            eval_documents=eval_documents,
        )

    if corpus == "file":
        if corpus_path is None:
            parser.error("--corpus-path is required when --corpus=file")
        try:
            documents = _load_documents(
                corpus_path,
                loader=document_loader,
                custom_loader=custom_loader,
                corpus_format=corpus_format,
                render_format=render_format,
            )
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
    distractor_policy: str = "hard",
) -> object:
    from intrep.next_observation_evaluation import (
        evaluate_next_observation_learning as evaluate,
    )

    return evaluate(
        documents,
        eval_documents=eval_documents,
        training_config=training_config,
        distractor_policy=distractor_policy,
    )


def _write_metrics(
    path: Path,
    result: object,
    corpus_label: str,
    eval_label: str,
    distractor_policy: str,
    training_config: GPTTrainingConfig,
) -> None:
    before_metrics = getattr(result, "before_metrics")
    after_metrics = getattr(result, "after_metrics")
    eval_split, generalization_eval = _eval_reporting(eval_label)
    warnings = []
    if not generalization_eval:
        warnings.append(
            "No held-out eval corpus was provided; ranking evaluation uses train cases and is not a generalization estimate."
        )
    payload = {
        "corpus": corpus_label,
        "eval_corpus": eval_label,
        "distractor_policy": distractor_policy,
        "eval_split": eval_split,
        "generalization_eval": generalization_eval,
        "warnings": warnings,
        "train_case_count": len(getattr(result, "train_cases")),
        "eval_case_count": len(getattr(result, "eval_cases")),
        "training": _training_result_to_dict(
            getattr(result, "training_result"),
            training_config,
            eval_split=eval_split,
            generalization_eval=generalization_eval,
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


def _eval_reporting(eval_label: str) -> tuple[str, bool]:
    if eval_label in ("train", "generated_seen"):
        return "train", False
    return "held_out", True


def _training_result_to_dict(
    result: object,
    training_config: GPTTrainingConfig,
    *,
    eval_split: str | None = None,
    generalization_eval: bool | None = None,
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
        "initial_step_loss": getattr(result, "initial_step_loss", getattr(result, "initial_loss", None)),
        "final_step_loss": getattr(result, "final_step_loss", getattr(result, "final_loss", None)),
        "best_loss": getattr(result, "best_loss", None),
        "best_step_loss": getattr(result, "best_step_loss", getattr(result, "best_loss", None)),
        "loss_reduction": getattr(result, "loss_reduction", None),
        "loss_reduction_ratio": getattr(result, "loss_reduction_ratio", None),
        "step_loss_reduction": getattr(result, "step_loss_reduction", getattr(result, "loss_reduction", None)),
        "step_loss_reduction_ratio": getattr(
            result,
            "step_loss_reduction_ratio",
            getattr(result, "loss_reduction_ratio", None),
        ),
        "initial_train_loss": getattr(result, "initial_train_loss", None),
        "final_train_loss": getattr(result, "final_train_loss", None),
        "initial_eval_loss": getattr(result, "initial_eval_loss", None),
        "final_eval_loss": getattr(result, "final_eval_loss", None),
        "eval_split": eval_split if eval_split is not None else getattr(result, "eval_split", None),
        "generalization_eval": (
            generalization_eval
            if generalization_eval is not None
            else getattr(result, "generalization_eval", None)
        ),
        "warnings": list(getattr(result, "warnings", ())),
    }


def _ranking_summary_to_dict(
    result: object,
    prefix: str,
    fallback_metrics: object,
) -> dict[str, object]:
    summary = getattr(result, f"{prefix}_summary", None)
    if summary is None:
        return {
            "overall": _ranking_metrics_to_dict(fallback_metrics),
            "fallback_counts": {},
        }
    return {
        "overall": _ranking_metrics_to_dict(getattr(summary, "overall")),
        "per_modality": {
            modality: _ranking_metrics_to_dict(metrics)
            for modality, metrics in getattr(summary, "per_modality").items()
        },
        "modality_counts": dict(getattr(summary, "modality_counts")),
        "fallback_counts": dict(getattr(summary, "fallback_counts", {})),
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
