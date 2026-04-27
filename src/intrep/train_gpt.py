from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import GPT_MODEL_PRESETS, build_gpt_config, gpt_config_to_dict
from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt
from intrep.language_modeling_metrics import language_modeling_metrics_from_training_result
from intrep.mixed_corpus import MixedDocument
from intrep.run_summary import build_run_summary, write_json


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


def _add_model_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-preset", choices=sorted(GPT_MODEL_PRESETS), default="small")
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)


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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="Training device. The default stays CPU-compatible; use auto or cuda on GPU hosts.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Write a final training checkpoint. Resume is intentionally not supported.",
    )
    _add_model_config_arguments(parser)
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
    parser.add_argument("--run-id")
    parser.add_argument(
        "--run-summary-path",
        type=Path,
        help="Write normalized run summary JSON.",
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
        device=args.device,
        checkpoint_path=args.checkpoint_path,
    )
    try:
        model_config = build_gpt_config(
            preset=args.model_preset,
            vocab_size=ByteTokenizer.vocab_size,
            context_length=args.context_length,
            embedding_dim=args.embedding_dim,
            num_heads=args.num_heads,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    except ValueError as error:
        parser.error(str(error))
    started_at = time.perf_counter()
    try:
        result = train_mixed_gpt(
            documents=documents,
            eval_documents=eval_documents,
            training_config=training_config,
            model_config=model_config,
        )
    except ValueError as error:
        parser.error(str(error))
    elapsed_seconds = time.perf_counter() - started_at
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
        f" device={getattr(result, 'device', training_config.device)}"
        f" model={json.dumps(gpt_config_to_dict(model_config), sort_keys=True)}"
    )
    if args.loss_summary:
        print(_loss_summary(result))
    if args.loss_history_path is not None:
        _write_loss_history(args.loss_history_path, result, training_config)
    if args.run_summary_path is not None:
        write_json(
            args.run_summary_path,
            build_run_summary(
                kind="train_gpt",
                run_id=args.run_id,
                corpus={
                    "train": {"label": corpus_label, "path": str(args.corpus_path) if args.corpus_path else None},
                    "eval": {"label": eval_label, "path": str(args.eval_corpus_path) if args.eval_corpus_path else None},
                },
                training_config=training_config,
                model_config=model_config,
                training_loss={
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "steps": result.steps,
                    "token_count": result.token_count,
                    "best_loss": result.best_loss,
                    "loss_reduction": result.loss_reduction,
                    "loss_reduction_ratio": result.loss_reduction_ratio,
                    "loss_history": list(result.loss_history),
                    "initial_train_loss": result.initial_train_loss,
                    "final_train_loss": result.final_train_loss,
                    "initial_eval_loss": result.initial_eval_loss,
                    "final_eval_loss": result.final_eval_loss,
                },
                language_modeling=language_modeling_metrics_from_training_result(result),
                elapsed_seconds=elapsed_seconds,
            ),
        )


if __name__ == "__main__":
    main()
