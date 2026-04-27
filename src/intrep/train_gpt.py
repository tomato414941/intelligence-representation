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
    return (
        f"loss initial={initial_loss:.4f}"
        f" final={final_loss:.4f}"
        f" best={best_loss:.4f}"
        f" delta={loss_reduction:.4f}"
        f" ratio={loss_reduction_ratio:.2%}"
    )


def _write_loss_history(path: Path, result: object) -> None:
    payload = {
        "steps": getattr(result, "steps"),
        "token_count": getattr(result, "token_count"),
        "initial_loss": getattr(result, "initial_loss"),
        "final_loss": getattr(result, "final_loss"),
        "best_loss": getattr(result, "best_loss"),
        "loss_history": list(getattr(result, "loss_history")),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a tiny decoder-only GPT on mixed-world data.")
    parser.add_argument(
        "--corpus",
        choices=("builtin", "file"),
        default="builtin",
        help="Use the built-in mixed-world corpus or load documents from a JSONL file.",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        help="Path to a mixed-document JSONL corpus when --corpus=file.",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
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
    corpus_label = "builtin"
    if args.corpus == "file":
        if args.corpus_path is None:
            parser.error("--corpus-path is required when --corpus=file")
        loader = document_loader or _load_file_documents
        try:
            documents = loader(args.corpus_path)
        except RuntimeError as error:
            parser.error(str(error))
        except (OSError, ValueError) as error:
            parser.error(f"could not load corpus from {args.corpus_path}: {error}")
        corpus_label = str(args.corpus_path)
    elif args.corpus_path is not None:
        parser.error("--corpus-path can only be used with --corpus=file")

    result = train_mixed_gpt(
        documents=documents,
        training_config=GPTTrainingConfig(
            max_steps=args.max_steps,
            context_length=args.context_length,
            batch_size=args.batch_size,
        )
    )
    print("intrep mixed-gpt training")
    print(
        f"corpus={corpus_label}"
        f" tokens={result.token_count}"
        f" steps={result.steps}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
    )
    if args.loss_summary:
        print(_loss_summary(result))
    if args.loss_history_path is not None:
        _write_loss_history(args.loss_history_path, result)


if __name__ == "__main__":
    main()
