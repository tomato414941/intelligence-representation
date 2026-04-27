from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt
from intrep.mixed_corpus import MixedDocument


DocumentLoader = Callable[[str | Path], list[MixedDocument]]


def _load_file_documents(path: str | Path) -> list[MixedDocument]:
    try:
        from intrep.mixed_corpus import load_mixed_documents_jsonl
    except ImportError as error:
        raise RuntimeError("file-backed corpus loading is not available in this build") from error
    return load_mixed_documents_jsonl(path)


def _load_builtin_grid_documents() -> list[MixedDocument]:
    from intrep import grid_corpus

    default_grid_documents = grid_corpus.default_grid_documents
    return default_grid_documents()


def _loss_summary(result: object) -> str:
    initial_loss = getattr(result, "initial_loss")
    final_loss = getattr(result, "final_loss")
    best_loss = getattr(result, "best_loss", final_loss)
    loss_reduction = getattr(result, "loss_reduction", initial_loss - final_loss)
    loss_reduction_ratio = getattr(
        result,
        "loss_reduction_ratio",
        0.0 if initial_loss == 0.0 else loss_reduction / initial_loss,
    )
    summary = (
        f"loss initial={initial_loss:.4f}"
        f" final={final_loss:.4f}"
        f" best={best_loss:.4f}"
        f" delta={loss_reduction:.4f}"
        f" ratio={loss_reduction_ratio:.2%}"
    )
    initial_eval_loss = getattr(result, "initial_eval_loss", None)
    final_eval_loss = getattr(result, "final_eval_loss", None)
    if initial_eval_loss is not None and final_eval_loss is not None:
        summary += f" eval_initial={initial_eval_loss:.4f} eval_final={final_eval_loss:.4f}"
    initial_train_loss = getattr(result, "initial_train_loss", None)
    final_train_loss = getattr(result, "final_train_loss", None)
    if initial_train_loss is not None and final_train_loss is not None:
        summary += f" train_avg_initial={initial_train_loss:.4f} train_avg_final={final_train_loss:.4f}"
    return summary


def _write_loss_history(path: Path, result: object, training_config: GPTTrainingConfig) -> None:
    payload = {
        "steps": getattr(result, "steps"),
        "token_count": getattr(result, "token_count"),
        "batch_stride": getattr(result, "batch_stride", training_config.batch_stride),
        "initial_loss": getattr(result, "initial_loss"),
        "final_loss": getattr(result, "final_loss"),
        "best_loss": getattr(result, "best_loss"),
        "loss_history": list(getattr(result, "loss_history")),
        "initial_train_loss": getattr(result, "initial_train_loss", None),
        "final_train_loss": getattr(result, "final_train_loss", None),
        "initial_eval_loss": getattr(result, "initial_eval_loss", None),
        "final_eval_loss": getattr(result, "final_eval_loss", None),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only GPT on mixed-world data.")
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
        help="Optional mixed-document JSONL corpus for held-out loss evaluation.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument(
        "--loss-summary",
        action="store_true",
        help="Print a compact one-line loss summary.",
    )
    parser.add_argument(
        "--loss-history-path",
        type=Path,
        help="Write training loss history and summary metrics to a JSON file.",
    )
    return parser


def main(argv: list[str] | None = None, document_loader: DocumentLoader | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    documents = None
    eval_documents = None
    corpus_label = "builtin"
    eval_label = "none"
    loader = document_loader or _load_file_documents
    if args.corpus == "builtin-grid":
        documents = _load_builtin_grid_documents()
        corpus_label = "builtin-grid"
    elif args.corpus == "file":
        if args.corpus_path is None:
            parser.error("--corpus-path is required when --corpus=file")
        try:
            documents = loader(args.corpus_path)
        except RuntimeError as error:
            parser.error(str(error))
        except (OSError, ValueError) as error:
            parser.error(f"could not load corpus from {args.corpus_path}: {error}")
        corpus_label = str(args.corpus_path)
    elif args.corpus_path is not None:
        parser.error("--corpus-path can only be used with --corpus=file")

    if args.eval_corpus_path is not None:
        try:
            eval_documents = loader(args.eval_corpus_path)
        except RuntimeError as error:
            parser.error(str(error))
        except (OSError, ValueError) as error:
            parser.error(f"could not load eval corpus from {args.eval_corpus_path}: {error}")
        eval_label = str(args.eval_corpus_path)

    training_config = GPTTrainingConfig(
        max_steps=args.max_steps,
        context_length=args.context_length,
        batch_size=args.batch_size,
        batch_stride=args.batch_stride,
    )
    result = train_mixed_gpt(
        documents=documents,
        eval_documents=eval_documents,
        training_config=training_config,
    )
    print("intrep mixed-gpt training")
    print(
        f"corpus={corpus_label}"
        f" eval_corpus={eval_label}"
        f" tokens={result.token_count}"
        f" steps={result.steps}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" train_avg_initial={result.initial_train_loss:.4f}"
        f" train_avg_final={result.final_train_loss:.4f}"
    )
    if args.loss_summary:
        print(_loss_summary(result))
    if args.loss_history_path is not None:
        _write_loss_history(args.loss_history_path, result, training_config)


if __name__ == "__main__":
    main()
