from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

from intrep.tasks.shogi_policy_value.checkpoint import load_shogi_policy_value_checkpoint
from intrep.tasks.shogi_policy_value.dataset_definition import (
    load_shogi_policy_value_dataset_definition,
    load_shogi_policy_value_dataset_examples,
    shogi_policy_value_dataset_definition_to_json,
)
from intrep.tasks.shogi_policy_value.examples import ShogiPolicyValueDataset, ShogiPolicyValueExample
from intrep.tasks.shogi_policy_value.training import evaluate_shogi_policy_value_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a shogi move-choice checkpoint without training.")
    parser.add_argument("--dataset-definition", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()

    payload = evaluate_shogi_policy_value_checkpoint(
        dataset_definition_path=args.dataset_definition,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        device=args.device,
        max_train_examples=args.max_train_examples,
        max_eval_examples=args.max_eval_examples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def evaluate_shogi_policy_value_checkpoint(
    *,
    dataset_definition_path: Path,
    checkpoint_path: Path,
    batch_size: int = 128,
    device: str = "cpu",
    max_train_examples: int | None = None,
    max_eval_examples: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    dataset_definition = load_shogi_policy_value_dataset_definition(dataset_definition_path)
    train_examples, eval_examples = load_shogi_policy_value_dataset_examples(dataset_definition)
    used_train_examples = _limit_examples(train_examples, max_train_examples, label="max train examples")
    used_eval_examples = _limit_examples(eval_examples, max_eval_examples, label="max eval examples")
    model = load_shogi_policy_value_checkpoint(checkpoint_path, device=device)
    train_metrics = evaluate_shogi_policy_value_metrics(
        model,
        _loader(used_train_examples, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
    )
    eval_metrics = evaluate_shogi_policy_value_metrics(
        model,
        _loader(used_eval_examples, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory),
    )
    return {
        "raw_train_case_count": len(train_examples),
        "raw_eval_case_count": len(eval_examples),
        "used_train_case_count": len(used_train_examples),
        "used_eval_case_count": len(used_eval_examples),
        "train_policy_target_summary": _policy_target_summary(train_examples),
        "eval_policy_target_summary": _policy_target_summary(eval_examples),
        "dataset_definition_path": str(dataset_definition_path),
        "dataset_definition": shogi_policy_value_dataset_definition_to_json(dataset_definition),
        "checkpoint_path": str(checkpoint_path),
        "batch_size": batch_size,
        "device": device,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "train_metrics": asdict(train_metrics),
        "eval_metrics": asdict(eval_metrics),
    }


def _loader(
    examples: list[ShogiPolicyValueExample],
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        ShogiPolicyValueDataset(examples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _limit_examples(
    examples: list[ShogiPolicyValueExample],
    max_examples: int | None,
    *,
    label: str,
) -> list[ShogiPolicyValueExample]:
    if max_examples is None:
        return examples
    if max_examples <= 0:
        raise ValueError(f"{label} must be positive")
    return examples[:max_examples]


def _policy_target_summary(examples: list[ShogiPolicyValueExample]) -> dict[str, float | int]:
    available_counts = [
        sum(1 for weight in example.policy_targets.values() if weight > 0.0)
        for example in examples
        if example.policy_targets is not None
    ]
    available_count = len(available_counts)
    total_count = len(examples)
    return {
        "available_count": available_count,
        "missing_count": total_count - available_count,
        "available_ratio": available_count / total_count if total_count else 0.0,
        "mean_nonzero_count": sum(available_counts) / available_count if available_count else 0.0,
    }


if __name__ == "__main__":
    main()
