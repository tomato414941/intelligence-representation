from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, overload

import torch

from intrep.problems.shogi_policy_value.data import (
    load_shogi_engine_analysis_by_position_jsonl,
    load_shogi_move_policy_value_examples_from_game_records_jsonl_with_engine_analysis,
)
from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    ShogiPolicyValueDataSelectionSource,
    load_shogi_policy_value_data_selection,
    shogi_policy_value_data_selection_to_json,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiMovePolicyValueExample,
    CandidateMovePolicyValueTensorSample,
    load_shogi_move_policy_value_examples_jsonl,
    tensorize_candidate_move_policy_value_examples,
)
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis
from intrep.worlds.shogi.position_encoding import SHOGI_POSITION_INPUT_ENCODING

SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA = "intrep.shogi_policy_value_tensor_cache.v2"
SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA = "intrep.shogi_policy_value_tensor_cache_shard.v1"
DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME = "shogi-policy-value-tensors"


@dataclass(frozen=True)
class ShogiPolicyValueTensorCache:
    train_samples: Sequence[CandidateMovePolicyValueTensorSample]
    eval_samples: Sequence[CandidateMovePolicyValueTensorSample]
    train_policy_target_summary: dict[str, float | int]
    eval_policy_target_summary: dict[str, float | int]


def default_shogi_policy_value_tensor_cache_path(data_selection_path: Path) -> Path:
    return data_selection_path.parent / "cache" / DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME


def build_shogi_policy_value_tensor_cache(
    *,
    data_selection_path: Path,
    output_path: Path | None = None,
    shard_examples: int = 100_000,
    shard_games: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    if shard_games is not None:
        shard_examples = shard_games
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    cache_dir = output_path or default_shogi_policy_value_tensor_cache_path(data_selection_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    train_summary = _empty_policy_target_summary()
    eval_summary = _empty_policy_target_summary()
    max_choice_count = 0
    analyses_by_position = load_shogi_engine_analysis_by_position_jsonl(
        tuple(source.path for source in data_selection.analysis_sources)
    )

    for split, sources in (("train", data_selection.train_sources), ("eval", data_selection.eval_sources)):
        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for source_index, source in enumerate(sources):
            for shard in _build_source_shards(
                split=split,
                source=source,
                source_index=source_index,
                split_dir=split_dir,
                shard_examples=shard_examples,
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
        "input_encoding": SHOGI_POSITION_INPUT_ENCODING,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection, root=data_selection_path.parent),
        "shard_examples": shard_examples,
        "train_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "train"),
        "eval_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "eval"),
        "skipped_example_count": sum(int(shard.get("skipped_example_count", 0)) for shard in shards),
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
        "skipped_example_count": manifest["skipped_example_count"],
        "shard_count": len(shards),
    }


def build_shogi_policy_value_tensor_cache_shard(
    *,
    data_selection_path: Path,
    cache_dir: Path,
    split: str,
    source_index: int,
    source_example_start_index: int | None = None,
    source_example_end_index: int | None = None,
    source_game_start_index: int | None = None,
    source_game_end_index: int | None = None,
    shard_index: int,
    resume: bool = False,
) -> dict[str, object]:
    if source_example_start_index is None:
        source_example_start_index = source_game_start_index
    if source_example_end_index is None:
        source_example_end_index = source_game_end_index
    if source_example_start_index is None or source_example_end_index is None:
        raise ValueError("source_example_start_index and source_example_end_index are required")
    if source_example_start_index < 0:
        raise ValueError("source_example_start_index must be non-negative")
    if source_example_end_index <= source_example_start_index:
        raise ValueError("source_example_end_index must be greater than source_example_start_index")
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    sources = _split_sources(data_selection, split)
    if source_index < 0 or source_index >= len(sources):
        raise ValueError("source_index is out of range")
    source = sources[source_index]
    analyses_by_position = load_shogi_engine_analysis_by_position_jsonl(
        tuple(source.path for source in data_selection.analysis_sources)
    )
    source_examples = _source_examples(source, data_selection=data_selection, analyses_by_position=analyses_by_position)
    examples = [
        (source_example_index, example)
        for source_example_index, example in enumerate(source_examples)
        if source_example_start_index <= source_example_index < source_example_end_index
    ]
    if not examples:
        raise ValueError("shard range must contain at least one example")
    split_dir = cache_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    return _build_shard(
        split=split,
        source=source,
        source_index=source_index,
        split_dir=split_dir,
        shard_index=shard_index,
        examples=examples,
        data_selection=data_selection,
        data_selection_path=data_selection_path,
        resume=resume,
    )


def write_shogi_policy_value_tensor_cache_manifest(
    *,
    data_selection_path: Path,
    cache_dir: Path,
    shard_examples: int = 100_000,
    shard_games: int | None = None,
) -> dict[str, object]:
    if shard_games is not None:
        shard_examples = shard_games
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    shards = sorted(
        (_load_shard_manifest_file(path) for path in cache_dir.glob("*/*.json")),
        key=lambda shard: (
            str(shard["split"]),
            int(shard["source_index"]),
            int(shard["source_example_start_index"]),
        ),
    )
    if not shards:
        raise ValueError("tensor cache must contain at least one shard manifest")
    train_summary = _empty_policy_target_summary()
    eval_summary = _empty_policy_target_summary()
    max_choice_count = 0
    for shard in shards:
        if shard.get("data_selection_path") != str(data_selection_path):
            raise ValueError(f"tensor cache shard data_selection_path does not match: {shard['path']}")
        max_choice_count = max(max_choice_count, int(shard["max_choice_count"]))
        summary = train_summary if shard["split"] == "train" else eval_summary
        _merge_policy_target_summary(summary, _object_dict(shard["policy_target_summary"]))

    manifest = {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "input_encoding": SHOGI_POSITION_INPUT_ENCODING,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection, root=data_selection_path.parent),
        "shard_examples": shard_examples,
        "train_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "train"),
        "eval_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "eval"),
        "skipped_example_count": sum(int(shard.get("skipped_example_count", 0)) for shard in shards),
        "max_choice_count": max_choice_count,
        "train_policy_target_summary": _finalize_policy_target_summary(train_summary),
        "eval_policy_target_summary": _finalize_policy_target_summary(eval_summary),
        "shards": shards,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_shogi_policy_value_tensor_cache(
    path: Path,
    *,
    expected_data_selection: ShogiPolicyValueDataSelection | None = None,
    expected_data_selection_root: Path | None = None,
) -> ShogiPolicyValueTensorCache:
    manifest_path = path / "manifest.json"
    manifest = _object_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache schema")
    if manifest.get("input_encoding") != SHOGI_POSITION_INPUT_ENCODING:
        raise ValueError("unsupported shogi policy/value tensor cache input encoding")
    if expected_data_selection is not None:
        expected = shogi_policy_value_data_selection_to_json(
            expected_data_selection,
            root=expected_data_selection_root,
        )
        manifest_data_selection = _portable_manifest_data_selection(manifest)
        if manifest_data_selection != expected:
            raise ValueError("tensor cache data selection does not match requested data selection")
    train_shards = [_object_dict(shard) for shard in _object_list(manifest["shards"]) if _object_dict(shard)["split"] == "train"]
    eval_shards = [_object_dict(shard) for shard in _object_list(manifest["shards"]) if _object_dict(shard)["split"] == "eval"]
    max_choice_count = int(manifest["max_choice_count"])
    return ShogiPolicyValueTensorCache(
        train_samples=ShardedCandidateMovePolicyValueTensorSamples(path, train_shards, max_choice_count=max_choice_count),
        eval_samples=ShardedCandidateMovePolicyValueTensorSamples(path, eval_shards, max_choice_count=max_choice_count),
        train_policy_target_summary=_object_dict(manifest["train_policy_target_summary"]),
        eval_policy_target_summary=_object_dict(manifest["eval_policy_target_summary"]),
    )


def _portable_manifest_data_selection(manifest: dict[str, object]) -> dict[str, object]:
    data_selection = _object_dict(manifest["data_selection"])
    data_selection_path = manifest.get("data_selection_path")
    if not isinstance(data_selection_path, str) or not data_selection_path:
        return data_selection
    return _data_selection_json_with_paths_relative_to(data_selection, root=Path(data_selection_path).parent)


def _data_selection_json_with_paths_relative_to(payload: dict[str, object], *, root: Path) -> dict[str, object]:
    result = dict(payload)
    for key in ("train_sources", "eval_sources"):
        sources = []
        for source in _object_list(result.get(key, [])):
            source_payload = dict(_object_dict(source))
            source_path = Path(str(source_payload["path"]))
            if source_path.is_absolute():
                try:
                    source_path = source_path.relative_to(root)
                except ValueError:
                    pass
            source_payload["path"] = str(source_path)
            sources.append(source_payload)
        result[key] = sources
    return result


def _split_sources(
    data_selection: ShogiPolicyValueDataSelection,
    split: str,
) -> tuple[ShogiPolicyValueDataSelectionSource, ...]:
    if split == "train":
        return data_selection.train_sources
    if split == "eval":
        return data_selection.eval_sources
    raise ValueError("split must be train or eval")


class ShardedCandidateMovePolicyValueTensorSamples(Sequence[CandidateMovePolicyValueTensorSample]):
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
        self._loaded_samples: list[CandidateMovePolicyValueTensorSample] = []

    def __len__(self) -> int:
        return self.sample_count

    @overload
    def __getitem__(self, index: int) -> CandidateMovePolicyValueTensorSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[CandidateMovePolicyValueTensorSample]:
        ...

    def __getitem__(self, index: int | slice) -> CandidateMovePolicyValueTensorSample | list[CandidateMovePolicyValueTensorSample]:
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

    def _load_shard_samples(self, shard_index: int) -> list[CandidateMovePolicyValueTensorSample]:
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
    shard_examples: int,
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    resume: bool,
) -> list[dict[str, object]]:
    shards: list[dict[str, object]] = []
    batch: list[tuple[int, ShogiMovePolicyValueExample]] = []
    emitted = 0
    for source_example_index, example in enumerate(
        _source_examples(source, data_selection=data_selection, analyses_by_position=analyses_by_position)
    ):
        if source.max_examples is not None and emitted >= source.max_examples:
            break
        batch.append((source_example_index, example))
        emitted += 1
        if len(batch) >= shard_examples:
            shards.append(
                _build_shard(
                    split=split,
                    source=source,
                    source_index=source_index,
                    split_dir=split_dir,
                    shard_index=len(shards),
                    examples=batch,
                    data_selection=data_selection,
                    data_selection_path=data_selection_path,
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
                examples=batch,
                data_selection=data_selection,
                data_selection_path=data_selection_path,
                resume=resume,
            )
        )
    return shards


def _source_examples(
    source: ShogiPolicyValueDataSelectionSource,
    *,
    data_selection: ShogiPolicyValueDataSelection,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
) -> list[ShogiMovePolicyValueExample]:
    if source.kind == "shogi_policy_value_examples_jsonl":
        return load_shogi_move_policy_value_examples_jsonl(source.path, max_examples=source.max_examples)
    if source.kind == "game_records_jsonl":
        if data_selection.target_construction is None:
            raise ValueError("target_construction is required for game_records_jsonl sources")
        return load_shogi_move_policy_value_examples_from_game_records_jsonl_with_engine_analysis(
            source.path,
            policy_target_construction=data_selection.target_construction.policy,
            value_target_construction=data_selection.target_construction.value,
            analyses_by_position=analyses_by_position,
            policy_temperature_cp=data_selection.target_construction.policy_temperature_cp,
            policy_mate_cp=data_selection.target_construction.policy_mate_cp,
            score_cp_scale=data_selection.target_construction.score_cp_scale,
            max_games=source.max_games,
        )
    raise ValueError(f"unsupported data selection source kind: {source.kind}")


def _build_shard(
    *,
    split: str,
    source: ShogiPolicyValueDataSelectionSource,
    source_index: int,
    split_dir: Path,
    shard_index: int,
    examples: Sequence[tuple[int, ShogiMovePolicyValueExample]],
    data_selection: ShogiPolicyValueDataSelection,
    data_selection_path: Path,
    resume: bool,
) -> dict[str, object]:
    first_index = examples[0][0]
    last_index = examples[-1][0]
    shard_path = split_dir / f"source-{source_index:04d}-examples-{first_index:08d}-{last_index + 1:08d}.pt"
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

    shard_examples = [example for _source_example_index, example in examples]
    samples = tensorize_candidate_move_policy_value_examples(shard_examples)
    summary = _policy_target_summary(shard_examples)
    max_choice_count = max((int(sample.candidate_move_features.shape[0]) for sample in samples), default=0)
    payload = {
        **expected,
        "sample_count": len(samples),
        "max_choice_count": max_choice_count,
        "policy_target_summary": summary,
        "skipped_example_count": 0,
        "failures": [],
        "samples": [_sample_to_payload(sample) for sample in samples],
    }
    torch.save(payload, shard_path)
    _write_shard_manifest_file(_shard_manifest_path(shard_path), _shard_manifest(payload))
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
        "input_encoding": SHOGI_POSITION_INPUT_ENCODING,
        "split": split,
        "source_index": source_index,
        "shard_index": shard_index,
        "source_kind": source.kind,
        "source_path": str(source.path),
        "source_max_examples": source.max_examples,
        "source_example_start_index": first_index,
        "source_example_end_index": last_index + 1,
        "data_selection_path": str(data_selection_path),
        "path": str(path.relative_to(path.parents[1])),
    }


def _try_load_matching_shard(shard_path: Path, expected: dict[str, object]) -> dict[str, object] | None:
    manifest_path = _shard_manifest_path(shard_path)
    if manifest_path.exists() and shard_path.exists():
        try:
            manifest = _load_shard_manifest_file(manifest_path)
        except Exception:  # noqa: BLE001
            return None
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        if int(manifest.get("sample_count", 0)) < 0:
            return None
        return manifest

    try:
        payload = _load_shard(shard_path)
    except Exception:  # noqa: BLE001
        return None
    for key, value in expected.items():
        if payload.get(key) != value:
            return None
    if int(payload.get("sample_count", 0)) < 0:
        return None
    manifest = _shard_manifest(payload)
    _write_shard_manifest_file(manifest_path, manifest)
    return manifest


def _shard_manifest_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".json")


def _load_shard_manifest_file(path: Path) -> dict[str, object]:
    payload = _object_dict(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache shard manifest schema")
    if payload.get("input_encoding") != SHOGI_POSITION_INPUT_ENCODING:
        raise ValueError("unsupported shogi policy/value tensor cache shard input encoding")
    return payload


def _write_shard_manifest_file(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_shard(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache shard schema")
    if payload.get("input_encoding") != SHOGI_POSITION_INPUT_ENCODING:
        raise ValueError("unsupported shogi policy/value tensor cache shard input encoding")
    return payload


def _shard_manifest(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "input_encoding",
            "split",
            "source_index",
            "shard_index",
            "source_kind",
            "source_path",
            "source_max_examples",
            "source_example_start_index",
            "source_example_end_index",
            "data_selection_path",
            "sample_count",
            "max_choice_count",
            "policy_target_summary",
            "skipped_example_count",
            "failures",
            "path",
        )
    }


def _sample_to_payload(sample: CandidateMovePolicyValueTensorSample) -> dict[str, torch.Tensor]:
    return {
        "position_token_ids": sample.position_token_ids,
        "candidate_move_features": sample.candidate_move_features,
        "label": sample.label,
        "policy_targets": sample.policy_targets,
        "value_target": sample.value_target,
    }


def _sample_from_payload(payload: Any) -> CandidateMovePolicyValueTensorSample:
    if not isinstance(payload, dict):
        raise ValueError("tensor cache sample must be a mapping")
    return CandidateMovePolicyValueTensorSample(
        position_token_ids=payload["position_token_ids"],
        candidate_move_features=payload["candidate_move_features"],
        label=payload["label"],
        policy_targets=payload["policy_targets"],
        value_target=payload["value_target"],
    )


def _policy_target_summary(examples: Sequence[ShogiMovePolicyValueExample]) -> dict[str, float | int]:
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
