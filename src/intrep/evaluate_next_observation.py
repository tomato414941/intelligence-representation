from __future__ import annotations

import argparse
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
    print(
        f"corpus={corpus.label}"
        f" eval_corpus={eval_label}"
        f" train_cases={len(result.train_cases)}"
        f" eval_cases={len(result.eval_cases)}"
        f" tokens={result.training_result.token_count}"
        f" steps={result.training_result.steps}"
        f" before_top1_accuracy={result.before_metrics.top1_accuracy:.4f}"
        f" after_top1_accuracy={result.after_metrics.top1_accuracy:.4f}"
        f" before_margin={result.before_metrics.mean_margin:.4f}"
        f" after_margin={result.after_metrics.mean_margin:.4f}"
    )


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


if __name__ == "__main__":
    main()
