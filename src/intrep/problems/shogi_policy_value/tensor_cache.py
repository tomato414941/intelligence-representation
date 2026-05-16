from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, overload

import torch

from intrep.problems.shogi_policy_value.data import (
    load_shogi_engine_analysis_by_position_jsonl,
    shogi_policy_value_examples_from_game_record,
)
from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    ShogiPolicyValueDataSelectionSource,
    load_shogi_policy_value_data_selection,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyValueExample,
    TensorizedShogiPolicyValueSample,
    tensorize_shogi_policy_value_examples,
)
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis
from intrep.worlds.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl

SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA = "intrep.shogi_policy_value_tensor_cache.v2"
SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA = "intrep.shogi_policy_value_tensor_cache_shard.v1"
DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME = "shogi-policy-value-tensors"


@dataclass(frozen=True)
class ShogiPolicyValueTensorCache:
    train_samples: Sequence[TensorizedShogiPolicyValueSample]
    eval_samples: Sequence[TensorizedShogiPolicyValueSample]
    train_policy_target_summary: dict[str, float | int]
    eval_policy_target_summary: dict[str, float | int]


def default_shogi_policy_value_tensor_cache_path(data_selection_path: Path) -> Path:
    return data_selection_path.parent / "cache" / DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME


def build_shogi_policy_value_tensor_cache(
    *,
    data_selection_path: Path,
    output_path: Path | None = None,
    shard_games: int = 100,
    resume: bool = False,
) -> dict[str, object]:
    if shard_games <= 0:
        raise ValueError("shard_games must be positive")
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    analyses_by_position = load_shogi_engine_analysis_by_position_jsonl(
        tuple(source.path for source in data_selection.analysis_sources)
    )
    cache_dir = output_path or default_shogi_policy_value_tensor_cache_path(data_selection_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    train_summary = _empty_policy_target_summary()
    eval_summary = _empty_policy_target_summary()
    max_choice_count = 0

    for split, sources in (("train", data_selection.train_sources), ("eval", data_selection.eval_sources)):
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source_index, source in enumerate(sources):
            for shard in _build_source_shards(
                split=split,
                source=source,
                source_index=source_index,
                split_dir=split_dir,
                shard_games=shard_games,
                data_selection=data_selection,
                data_selection_path=data_selection_path,
                analyses_by_position=analyses_by_position,
                resume=resume,
            ):
                shards.append(shard)
                max_choice_count = max(max_choice_count, int(shard["max_choice_count"]))
                summary = train_summary if split == "train" else eval_summary
                _merge_policy_target_summary(summary, _object_dict(shard["policy_target_summary"]))

    manifest = {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection),
        "shard_games": shard_games,
        "train_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "train"),
        "eval_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "eval"),
        "max_choice_count": max_choice_count,
        "train_policy_target_summary": _finalize_policy_target_summary(train_summary),
        "eval_policy_target_summary": _finalize_policy_target_summary(eval_summary),
        "shards": shards,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "path": str(cache_dir),
        "data_selection_path": str(data_selection_path),
        "train_count": manifest["train_count"],
        "eval_count": manifest["eval_count"],
        "shard_count": len(shards),
    }


def load_shogi_policy_value_tensor_cache(
    path: Path,
    *,
    expected_data_selection: ShogiPolicyValueDataSelection | None = None,
) -> ShogiPolicyValueTensorCache:
    manifest_path = path / "manifest.json"
    manifest = _object_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache schema")
    if expected_data_selection is not None:
        expected = shogi_policy_value_data_selection_to_json(expected_data_selection)
        if manifest.get("data_selection") != expected:
            raise ValueError("tensor cache data selection does not match requested data selection")
    train_shards = [_object_dict(shard) for shard in _object_list(manifest["shards"]) if _object_dict(shard)["split"] == "train"]
    eval_shards = [_object_dict(shard) for shard in _object_list(manifest["shards"]) if _object_dict(shard)["split"] == "eval"]
    max_choice_count = int(manifest["max_choice_count"])
    return ShogiPolicyValueTensorCache(
        train_samples=ShardedShogiPolicyValueTensorSamples(path, train_shards, max_choice_count=max_choice_count),
        eval_samples=ShardedShogiPolicyValueTensorSamples(path, eval_shards, max_choice_count=max_choice_count),
        train_policy_target_summary=_object_dict(manifest["train_policy_target_summary"]),
        eval_policy_target_summary=_object_dict(manifest["eval_policy_target_summary"]),
    )


class ShardedShogiPolicyValueTensorSamples(Sequence[TensorizedShogiPolicyValueSample]):
    def __init__(self, cache_dir: Path, shards: Sequence[dict[str, object]], *, max_choice_count: int) -> None:
        self.cache_dir = cache_dir
        self.shards = tuple(shards)
        self.max_choice_count = max_choice_count
        self.offsets: list[int] = []
        self.sequential_access_preferred = True
        offset = 0
        for shard in self.shards:
            self.offsets.append(offset)
            offset += int(shard["sample_count"])
        self.sample_count = offset
        self._loaded_shard_index: int | None = None
        self._loaded_samples: list[TensorizedShogiPolicyValueSample] = []

    def __len__(self) -> int:
        return self.sample_count

    @overload
    def __getitem__(self, index: int) -> TensorizedShogiPolicyValueSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[TensorizedShogiPolicyValueSample]:
        ...

    def __getitem__(self, index: int | slice) -> TensorizedShogiPolicyValueSample | list[TensorizedShogiPolicyValueSample]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = self._shard_index_for_sample(index)
        shard_offset = self.offsets[shard_index]
        return self._load_shard_samples(shard_index)[index - shard_offset]

    def _shard_index_for_sample(self, index: int) -> int:
        low = 0
        high = len(self.shards) - 1
        while low <= high:
            mid = (low + high) // 2
            start = self.offsets[mid]
            end = start + int(self.shards[mid]["sample_count"])
            if index < start:
                high = mid - 1
            elif index >= end:
                low = mid + 1
            else:
                return mid
        raise IndexError(index)

    def _load_shard_samples(self, shard_index: int) -> list[TensorizedShogiPolicyValueSample]:
        if self._loaded_shard_index == shard_index:
            return self._loaded_samples
        shard = self.shards[shard_index]
        payload = _load_shard(self.cache_dir / str(shard["path"]))
        self._loaded_samples = [_sample_from_payload(item) for item in payload["samples"]]
        self._loaded_shard_index = shard_index
        return self._loaded_samples


def _build_source_shards(
    *,
    split: str,
    source: ShogiPolicyValueDataSelectionSource,
    source_index: int,
    split_dir: Path,
    shard_games: int,
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    resume: bool,
) -> list[dict[str, object]]:
    shards: list[dict[str, object]] = []
    batch: list[tuple[int, ShogiGameRecord]] = []
    emitted = 0
    for source_game_index, record in enumerate(iter_shogi_game_records_jsonl(source.path)):
        if source.max_games is not None and emitted >= source.max_games:
            break
        batch.append((source_game_index, record))
        emitted += 1
        if len(batch) >= shard_games:
            shards.append(
                _build_shard(
                    split=split,
                    source=source,
                    source_index=source_index,
                    split_dir=split_dir,
                    shard_index=len(shards),
                    records=batch,
                    data_selection=data_selection,
                    data_selection_path=data_selection_path,
                    analyses_by_position=analyses_by_position,
                    resume=resume,
                )
            )
            batch = []
    if batch:
        shards.append(
            _build_shard(
                split=split,
                source=source,
                source_index=source_index,
                split_dir=split_dir,
                shard_index=len(shards),
                records=batch,
                data_selection=data_selection,
                data_selection_path=data_selection_path,
                analyses_by_position=analyses_by_position,
                resume=resume,
            )
        )
    return shards


def _build_shard(
    *,
    split: str,
    source: ShogiPolicyValueDataSelectionSource,
    source_index: int,
    split_dir: Path,
    shard_index: int,
    records: Sequence[tuple[int, ShogiGameRecord]],
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    resume: bool,
) -> dict[str, object]:
    first_index = records[0][0]
    last_index = records[-1][0]
    shard_path = split_dir / f"source-{source_index:04d}-games-{first_index:08d}-{last_index + 1:08d}.pt"
    expected = _shard_identity(
        split=split,
        source=source,
        source_index=source_index,
        shard_index=shard_index,
        first_index=first_index,
        last_index=last_index,
        data_selection=data_selection,
        data_selection_path=data_selection_path,
        path=shard_path,
    )
    if resume and shard_path.exists():
        loaded = _try_load_matching_shard(shard_path, expected)
        if loaded is not None:
            return loaded

    examples: list[ShogiPolicyValueExample] = []
    for _source_game_index, record in records:
        examples.extend(
            shogi_policy_value_examples_from_game_record(
                record,
                policy_target_construction=data_selection.target_construction.policy,
                value_target_construction=data_selection.target_construction.value,
                analyses_by_position=analyses_by_position,
                policy_temperature_cp=data_selection.target_construction.policy_temperature_cp,
                policy_mate_cp=data_selection.target_construction.policy_mate_cp,
                score_cp_scale=data_selection.target_construction.score_cp_scale,
            )
        )
    samples = tensorize_shogi_policy_value_examples(examples)
    summary = _policy_target_summary(examples)
    max_choice_count = max((int(sample.candidate_move_features.shape[0]) for sample in samples), default=0)
    payload = {
        **expected,
        "sample_count": len(samples),
        "max_choice_count": max_choice_count,
        "policy_target_summary": summary,
        "samples": [_sample_to_payload(sample) for sample in samples],
    }
    torch.save(payload, shard_path)
    return _shard_manifest(payload)


def _shard_identity(
    *,
    split: str,
    source: ShogiPolicyValueDataSelectionSource,
    source_index: int,
    shard_index: int,
    first_index: int,
    last_index: int,
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    path: Path,
) -> dict[str, object]:
    return {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA,
        "split": split,
        "source_index": source_index,
        "shard_index": shard_index,
        "source_kind": source.kind,
        "source_path": str(source.path),
        "source_max_games": source.max_games,
        "source_game_start_index": first_index,
        "source_game_end_index": last_index + 1,
        "data_selection_path": str(data_selection_path),
        "target_construction": shogi_policy_value_data_selection_to_json(data_selection)["target_construction"],
        "path": str(path.relative_to(path.parents[1])),
    }


def _try_load_matching_shard(shard_path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    try:
        payload = _load_shard(shard_path)
    except Exception:  # noqa: BLE001
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    if int(payload.get("sample_count", 0)) <= 0:
        return None
    return _shard_manifest(payload)


def _load_shard(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache shard schema")
    return payload


def _shard_manifest(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "split",
            "source_index",
            "shard_index",
            "source_kind",
            "source_path",
            "source_max_games",
            "source_game_start_index",
            "source_game_end_index",
            "sample_count",
            "max_choice_count",
            "policy_target_summary",
            "path",
        )
    }


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


def _policy_target_summary(examples: Sequence[ShogiPolicyValueExample]) -> dict[str, float | int]:
    summary = _empty_policy_target_summary()
    for example in examples:
        summary["total_count"] += 1
        if example.policy_targets is None:
            continue
        available_count = sum(1 for weight in example.policy_targets.values() if weight > 0.0)
        summary["available_count"] += 1
        summary["nonzero_count_sum"] += available_count
    return _finalize_policy_target_summary(summary)


def _empty_policy_target_summary() -> dict[str, float | int]:
    return {
        "total_count": 0,
        "available_count": 0,
        "nonzero_count_sum": 0,
    }


def _merge_policy_target_summary(target: dict[str, float | int], source: dict[str, object]) -> None:
    target["total_count"] += int(source["total_count"])
    target["available_count"] += int(source["available_count"])
    target["nonzero_count_sum"] += int(source["nonzero_count_sum"])


def _finalize_policy_target_summary(summary: dict[str, float | int]) -> dict[str, float | int]:
    total_count = int(summary["total_count"])
    available_count = int(summary["available_count"])
    nonzero_count_sum = int(summary["nonzero_count_sum"])
    return {
        "total_count": total_count,
        "available_count": available_count,
        "missing_count": total_count - available_count,
        "available_ratio": available_count / total_count if total_count else 0.0,
        "mean_nonzero_count": nonzero_count_sum / available_count if available_count else 0.0,
        "nonzero_count_sum": nonzero_count_sum,
    }


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value
