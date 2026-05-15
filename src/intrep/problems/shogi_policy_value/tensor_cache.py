from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    load_shogi_policy_value_data_selection,
    load_shogi_policy_value_data_selection_examples,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyValueExample,
    TensorizedShogiPolicyValueSample,
    tensorize_shogi_policy_value_examples,
)

SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA = "intrep.shogi_policy_value_tensor_cache.v1"
DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME = "shogi-policy-value-tensors.pt"


@dataclass(frozen=True)
class ShogiPolicyValueTensorCache:
    train_samples: list[TensorizedShogiPolicyValueSample]
    eval_samples: list[TensorizedShogiPolicyValueSample]
    train_policy_target_summary: dict[str, float | int]
    eval_policy_target_summary: dict[str, float | int]


def default_shogi_policy_value_tensor_cache_path(data_selection_path: Path) -> Path:
    return data_selection_path.parent / "cache" / DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME


def build_shogi_policy_value_tensor_cache(
    *,
    data_selection_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    train_examples, eval_examples = load_shogi_policy_value_data_selection_examples(data_selection)
    train_samples = tensorize_shogi_policy_value_examples(train_examples)
    eval_samples = tensorize_shogi_policy_value_examples(eval_examples)
    cache_path = output_path or default_shogi_policy_value_tensor_cache_path(data_selection_path)
    save_shogi_policy_value_tensor_cache(
        cache_path,
        data_selection=data_selection,
        data_selection_path=data_selection_path,
        train_samples=train_samples,
        eval_samples=eval_samples,
        train_policy_target_summary=_policy_target_summary(train_examples),
        eval_policy_target_summary=_policy_target_summary(eval_examples),
    )
    return {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "path": str(cache_path),
        "data_selection_path": str(data_selection_path),
        "train_count": len(train_samples),
        "eval_count": len(eval_samples),
    }


def save_shogi_policy_value_tensor_cache(
    path: Path,
    *,
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    train_samples: list[TensorizedShogiPolicyValueSample],
    eval_samples: list[TensorizedShogiPolicyValueSample],
    train_policy_target_summary: dict[str, float | int],
    eval_policy_target_summary: dict[str, float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection),
        "train_policy_target_summary": train_policy_target_summary,
        "eval_policy_target_summary": eval_policy_target_summary,
        "train_samples": [_sample_to_payload(sample) for sample in train_samples],
        "eval_samples": [_sample_to_payload(sample) for sample in eval_samples],
    }
    torch.save(payload, path)


def load_shogi_policy_value_tensor_cache(
    path: Path,
    *,
    expected_data_selection: ShogiPolicyValueDataSelection | None = None,
) -> ShogiPolicyValueTensorCache:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache schema")
    if expected_data_selection is not None:
        expected = shogi_policy_value_data_selection_to_json(expected_data_selection)
        if payload.get("data_selection") != expected:
            raise ValueError("tensor cache data selection does not match requested data selection")
    return ShogiPolicyValueTensorCache(
        train_samples=[_sample_from_payload(item) for item in payload["train_samples"]],
        eval_samples=[_sample_from_payload(item) for item in payload["eval_samples"]],
        train_policy_target_summary=dict(payload["train_policy_target_summary"]),
        eval_policy_target_summary=dict(payload["eval_policy_target_summary"]),
    )


def _sample_to_payload(sample: TensorizedShogiPolicyValueSample) -> dict[str, torch.Tensor]:
    return {
        "position_token_ids": sample.position_token_ids,
        "candidate_move_features": sample.candidate_move_features,
        "label": sample.label,
        "policy_targets": sample.policy_targets,
        "value_target": sample.value_target,
    }


def _sample_from_payload(payload: Any) -> TensorizedShogiPolicyValueSample:
    if not isinstance(payload, dict):
        raise ValueError("tensor cache sample must be a mapping")
    return TensorizedShogiPolicyValueSample(
        position_token_ids=payload["position_token_ids"],
        candidate_move_features=payload["candidate_move_features"],
        label=payload["label"],
        policy_targets=payload["policy_targets"],
        value_target=payload["value_target"],
    )


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
