from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.future_prediction_cases import FuturePredictionCase, extract_future_prediction_cases
from intrep.future_prediction_ranking import (
    FuturePredictionRankingMetrics,
    FuturePredictionRankingSummary,
    evaluate_future_prediction_ranking,
)
from intrep.gpt_model import DecoderOnlyGPT, build_gpt_config
from intrep.gpt_training import GPTTrainingConfig, train_mixed_gpt_with_artifacts
from intrep.signal_corpus import (
    load_signals_jsonl,
    reject_payload_refs,
    signals_to_mixed_documents,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate target-channel future prediction over signal streams."
    )
    parser.add_argument("--train-path", type=Path, required=True)
    parser.add_argument("--eval-path", type=Path)
    parser.add_argument(
        "--target-channel",
        choices=("consequence", "tool_result", "prediction_error"),
        default="consequence",
    )
    parser.add_argument(
        "--condition",
        choices=(
            "same_modality_negative",
            "same_action_different_context",
            "same_history_different_action",
        ),
        default="same_modality_negative",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-stride", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-preset", choices=("tiny", "small"), default="small")
    parser.add_argument(
        "--rendering",
        choices=("signal", "payload"),
        default="signal",
        help="Text rendering used for ranking prefixes and continuations.",
    )
    parser.add_argument("--metrics-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train_events = load_signals_jsonl(args.train_path)
        eval_events = load_signals_jsonl(args.eval_path) if args.eval_path else train_events
        reject_payload_refs(train_events, context="future prediction training")
        if args.eval_path:
            reject_payload_refs(eval_events, context="future prediction evaluation")
        train_cases = extract_future_prediction_cases(
            train_events,
            target_channel=args.target_channel,
            condition=args.condition,
        )
        eval_cases = extract_future_prediction_cases(
            eval_events,
            target_channel=args.target_channel,
            condition=args.condition,
        )
        if not train_cases:
            raise ValueError("training signals did not produce future prediction cases")
        if not eval_cases:
            raise ValueError("evaluation signals did not produce future prediction cases")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    training_config = GPTTrainingConfig(
        context_length=args.context_length,
        batch_stride=args.batch_stride,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    model_config = build_gpt_config(
        preset=args.model_preset,
        vocab_size=ByteTokenizer.vocab_size,
        context_length=args.context_length,
    )
    tokenizer = ByteTokenizer()
    torch.manual_seed(args.seed)
    before_model = DecoderOnlyGPT(model_config)
    before_summary = evaluate_future_prediction_ranking(
        eval_cases,
        before_model,
        tokenizer,
        rendering=args.rendering,
    )
    artifacts = train_mixed_gpt_with_artifacts(
        documents=signals_to_mixed_documents(train_events),
        eval_documents=signals_to_mixed_documents(eval_events) if args.eval_path else None,
        training_config=training_config,
        model_config=model_config,
    )
    after_summary = evaluate_future_prediction_ranking(
        eval_cases,
        artifacts.model,
        artifacts.tokenizer,
        rendering=args.rendering,
    )
    generalization_eval = args.eval_path is not None
    eval_split = "held_out" if generalization_eval else "train"
    print("intrep future-prediction evaluation")
    print(
        f"train_path={args.train_path}"
        f" eval_path={args.eval_path if args.eval_path else 'train'}"
        f" train_cases={len(train_cases)}"
        f" eval_cases={len(eval_cases)}"
        f" target_channel={args.target_channel}"
        f" condition={args.condition}"
        f" rendering={args.rendering}"
        f" eval_split={eval_split}"
        f" generalization_eval={str(generalization_eval).lower()}"
        f" before_top1_accuracy={before_summary.overall.top1_accuracy:.4f}"
        f" after_top1_accuracy={after_summary.overall.top1_accuracy:.4f}"
        f" before_margin={before_summary.overall.mean_margin:.4f}"
        f" after_margin={after_summary.overall.mean_margin:.4f}"
        f" train_final_loss={artifacts.result.final_train_loss:.4f}"
    )
    if args.metrics_path is not None:
        _write_metrics(
            args.metrics_path,
            before_summary=before_summary,
            after_summary=after_summary,
            eval_cases=eval_cases,
            train_case_count=len(train_cases),
            eval_case_count=len(eval_cases),
            target_channel=args.target_channel,
            condition=args.condition,
            rendering=args.rendering,
            eval_split=eval_split,
            generalization_eval=generalization_eval,
        )


def _write_metrics(
    path: Path,
    *,
    before_summary: FuturePredictionRankingSummary,
    after_summary: FuturePredictionRankingSummary,
    eval_cases: list[FuturePredictionCase],
    train_case_count: int,
    eval_case_count: int,
    target_channel: str,
    condition: str,
    rendering: str,
    eval_split: str,
    generalization_eval: bool,
) -> None:
    payload = {
        "target_channel": target_channel,
        "condition": condition,
        "rendering": rendering,
        "eval_split": eval_split,
        "generalization_eval": generalization_eval,
        "train_case_count": train_case_count,
        "eval_case_count": eval_case_count,
        "delta_top1_accuracy": after_summary.overall.top1_accuracy - before_summary.overall.top1_accuracy,
        "delta_margin": after_summary.overall.mean_margin - before_summary.overall.mean_margin,
        "explicit_negative_rate": _explicit_negative_rate(eval_cases),
        "no_negative_case_count": sum(1 for case in eval_cases if not case.negative_events),
        "ranking": {
            "before": _summary_to_dict(before_summary),
            "after": _summary_to_dict(after_summary),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _explicit_negative_rate(cases: list[FuturePredictionCase]) -> float:
    return 0.0


def _summary_to_dict(summary: FuturePredictionRankingSummary) -> dict[str, object]:
    return {
        "overall": _metrics_to_dict(summary.overall),
        "by_condition": {
            condition: _metrics_to_dict(metrics)
            for condition, metrics in summary.by_condition.items()
        },
        "condition_counts": dict(summary.condition_counts),
    }


def _metrics_to_dict(metrics: FuturePredictionRankingMetrics) -> dict[str, float]:
    return {
        "top1_accuracy": metrics.top1_accuracy,
        "mean_positive_loss": metrics.mean_positive_loss,
        "mean_best_negative_loss": metrics.mean_best_negative_loss,
        "mean_margin": metrics.mean_margin,
    }


if __name__ == "__main__":
    main()
