from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from intrep.byte_tokenizer import ByteTokenizer
from intrep.gpt_model import GPT_MODEL_PRESETS, build_gpt_config, gpt_config_to_dict
from intrep.gpt_training import GPTTrainingConfig, train_rendered_gpt_with_artifacts
from intrep.language_modeling_metrics import language_modeling_metrics_from_training_result
from intrep.run_summary import build_run_summary, write_json
from intrep.signal_io import load_signals_jsonl, reject_payload_refs
from intrep.signal_rendering import render_signals_for_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a text predictor on text-payload Signal JSONL."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=sorted(GPT_MODEL_PRESETS), default="small")
    parser.add_argument("--tokenizer", choices=("byte", "byte-pair"), default="byte")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=512)
    parser.add_argument("--tokenizer-min-pair-count", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--loss-summary", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-summary-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train_events = load_signals_jsonl(args.train_path)
        eval_events = load_signals_jsonl(args.eval_path) if args.eval_path is not None else None
        reject_payload_refs(train_events, context="signal text training")
        if eval_events is not None:
            reject_payload_refs(eval_events, context="signal text evaluation")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    training_config = GPTTrainingConfig(
        max_steps=args.max_steps,
        context_length=args.context_length,
        batch_size=args.batch_size,
        batch_stride=args.batch_stride,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        tokenizer=args.tokenizer,
        tokenizer_vocab_size=args.tokenizer_vocab_size,
        tokenizer_min_pair_count=args.tokenizer_min_pair_count,
    )
    model_vocab_size = (
        ByteTokenizer.vocab_size
        if args.tokenizer == "byte"
        else args.tokenizer_vocab_size
    )
    try:
        model_config = build_gpt_config(
            preset=args.model_preset,
            vocab_size=model_vocab_size,
            context_length=args.context_length,
        )
    except ValueError as error:
        parser.error(str(error))

    train_corpus = render_signals_for_training(train_events, render_format="signal-tags")
    eval_corpus = (
        render_signals_for_training(eval_events, render_format="signal-tags")
        if eval_events is not None
        else None
    )
    started_at = time.perf_counter()
    try:
        artifacts = train_rendered_gpt_with_artifacts(
            corpus=train_corpus,
            eval_corpus=eval_corpus,
            training_config=training_config,
            model_config=model_config,
        )
    except ValueError as error:
        parser.error(str(error))
    elapsed_seconds = time.perf_counter() - started_at
    result = artifacts.result
    eval_label = str(args.eval_path) if args.eval_path is not None else "none"
    print("intrep signal-text training")
    print(
        f"train_path={args.train_path}"
        f" eval_path={eval_label}"
        f" events={len(train_events)}"
        f" eval_events={len(eval_events) if eval_events is not None else 0}"
        f" tokens={result.token_count}"
        f" steps={result.steps}"
        f" initial_loss={result.initial_loss:.4f}"
        f" final_loss={result.final_loss:.4f}"
        f" tokenizer={args.tokenizer}"
        f" eval_split={result.eval_split}"
        f" generalization_eval={str(result.generalization_eval).lower()}"
        f" device={result.device}"
        f" model={json.dumps(gpt_config_to_dict(model_config), sort_keys=True)}"
    )
    if args.loss_summary:
        print(_loss_summary(result))
    if args.run_summary_path is not None:
        write_json(
            args.run_summary_path,
            build_run_summary(
                kind="train_signal_text",
                run_id=args.run_id,
                corpus={
                    "train": {"label": str(args.train_path), "path": str(args.train_path)},
                    "eval": {
                        "label": eval_label,
                        "path": str(args.eval_path) if args.eval_path else None,
                    },
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
                    "eval_split": result.eval_split,
                    "generalization_eval": result.generalization_eval,
                    "warnings": list(result.warnings),
                },
                language_modeling=language_modeling_metrics_from_training_result(result),
                elapsed_seconds=elapsed_seconds,
            ),
        )


def _loss_summary(result: object) -> str:
    initial_loss = getattr(result, "initial_loss")
    final_loss = getattr(result, "final_loss")
    loss_reduction = getattr(result, "loss_reduction", initial_loss - final_loss)
    loss_reduction_ratio = getattr(
        result,
        "loss_reduction_ratio",
        0.0 if initial_loss == 0.0 else loss_reduction / initial_loss,
    )
    return (
        f"loss initial={initial_loss:.4f}"
        f" final={final_loss:.4f}"
        f" delta={loss_reduction:.4f}"
        f" ratio={loss_reduction_ratio:.2%}"
    )


if __name__ == "__main__":
    main()
