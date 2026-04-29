from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from intrep.gpt_training import GPTTrainingConfig, train_rendered_gpt_with_artifacts
from intrep.signal_io import load_signals_jsonl, reject_payload_refs
from intrep.signal_rendering import render_signals_for_training


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
    parser.add_argument("--max-negatives", type=int)
    parser.add_argument("--metrics-path", type=Path)
    return parser


@dataclass(frozen=True)
class FuturePredictionEvaluationConfig:
    train_path: Path
    eval_path: Path | None = None
    target_channel: str = "consequence"
    condition: str = "same_modality_negative"
    max_steps: int = 20
    context_length: int = 64
    batch_size: int = 8
    batch_stride: int | None = None
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "small"
    rendering: str = "signal"
    image_patch_size: int = 1
    image_channel_bins: int = 4
    image_token_format: str = "flat"
    max_negatives: int | None = None
    metrics_path: Path | None = None


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = FuturePredictionEvaluationConfig(
        train_path=args.train_path,
        eval_path=args.eval_path,
        target_channel=args.target_channel,
        condition=args.condition,
        max_steps=args.max_steps,
        context_length=args.context_length,
        batch_size=args.batch_size,
        batch_stride=args.batch_stride,
        learning_rate=args.learning_rate,
        seed=args.seed,
        model_preset=args.model_preset,
        rendering=args.rendering,
        max_negatives=args.max_negatives,
        metrics_path=args.metrics_path,
    )
    try:
        run_future_prediction_evaluation(config)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def run_future_prediction_evaluation(config: FuturePredictionEvaluationConfig) -> None:
    train_events = load_signals_jsonl(config.train_path)
    eval_events = load_signals_jsonl(config.eval_path) if config.eval_path else train_events
    if config.rendering != "image-tokens":
        reject_payload_refs(train_events, context="future prediction training")
    if config.eval_path and config.rendering != "image-tokens":
        reject_payload_refs(eval_events, context="future prediction evaluation")
    train_cases = extract_future_prediction_cases(
        train_events,
        target_channel=config.target_channel,
        condition=config.condition,
    )
    eval_cases = extract_future_prediction_cases(
        eval_events,
        target_channel=config.target_channel,
        condition=config.condition,
    )
    if not train_cases:
        raise ValueError("training signals did not produce future prediction cases")
    if not eval_cases:
        raise ValueError("evaluation signals did not produce future prediction cases")

    training_config = GPTTrainingConfig(
        context_length=config.context_length,
        batch_stride=config.batch_stride,
        batch_size=config.batch_size,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    model_config = build_gpt_config(
        preset=config.model_preset,
        vocab_size=ByteTokenizer.vocab_size,
        context_length=config.context_length,
    )
    tokenizer = ByteTokenizer()
    torch.manual_seed(config.seed)
    before_model = DecoderOnlyGPT(model_config)
    before_summary = evaluate_future_prediction_ranking(
        eval_cases,
        before_model,
        tokenizer,
        rendering=config.rendering,
        image_patch_size=config.image_patch_size,
        image_channel_bins=config.image_channel_bins,
        image_token_format=config.image_token_format,
        max_negatives=config.max_negatives,
    )
    artifacts = train_rendered_gpt_with_artifacts(
        corpus=render_signals_for_training(
            train_events,
            render_format="image-tokens" if config.rendering == "image-tokens" else "signal-tags",
            image_patch_size=config.image_patch_size,
            image_channel_bins=config.image_channel_bins,
            image_token_format=config.image_token_format,
        ),
        eval_corpus=(
            render_signals_for_training(
                eval_events,
                render_format="image-tokens" if config.rendering == "image-tokens" else "signal-tags",
                image_patch_size=config.image_patch_size,
                image_channel_bins=config.image_channel_bins,
                image_token_format=config.image_token_format,
            )
            if config.eval_path
            else None
        ),
        training_config=training_config,
        model_config=model_config,
    )
    after_summary = evaluate_future_prediction_ranking(
        eval_cases,
        artifacts.model,
        artifacts.tokenizer,
        rendering=config.rendering,
        image_patch_size=config.image_patch_size,
        image_channel_bins=config.image_channel_bins,
        image_token_format=config.image_token_format,
        max_negatives=config.max_negatives,
    )
    generalization_eval = config.eval_path is not None
    eval_split = "held_out" if generalization_eval else "train"
    print("intrep future-prediction evaluation")
    print(
        f"train_path={config.train_path}"
        f" eval_path={config.eval_path if config.eval_path else 'train'}"
        f" train_cases={len(train_cases)}"
        f" eval_cases={len(eval_cases)}"
        f" target_channel={config.target_channel}"
        f" condition={config.condition}"
        f" rendering={config.rendering}"
        f" max_negatives={config.max_negatives if config.max_negatives is not None else 'all'}"
        f" eval_split={eval_split}"
        f" generalization_eval={str(generalization_eval).lower()}"
        f" before_top1_accuracy={before_summary.overall.top1_accuracy:.4f}"
        f" after_top1_accuracy={after_summary.overall.top1_accuracy:.4f}"
        f" before_margin={before_summary.overall.mean_margin:.4f}"
        f" after_margin={after_summary.overall.mean_margin:.4f}"
        f" train_final_loss={artifacts.result.final_train_loss:.4f}"
    )
    if config.metrics_path is not None:
        _write_metrics(
            config.metrics_path,
            before_summary=before_summary,
            after_summary=after_summary,
            eval_cases=eval_cases,
            train_case_count=len(train_cases),
            eval_case_count=len(eval_cases),
            target_channel=config.target_channel,
            condition=config.condition,
            rendering=config.rendering,
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
