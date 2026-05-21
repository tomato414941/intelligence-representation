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
    load_shogi_move_policy_value_examples_jsonl,
    shogi_move_policy_value_example_from_json,
)
from intrep.problems.shogi_policy_value.samples import (
    LegalMovePolicyValueTensorSample,
    CompactPolicyPlaneValueTensorSample,
)
from intrep.problems.shogi_policy_value.tensorization import (
    tensorize_legal_move_policy_value_examples,
    tensorize_compact_policy_plane_value_examples,
)
from intrep.problems.shogi_policy_value.output_space import (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
    validate_shogi_policy_value_output_space,
)
from intrep.problems.shogi_policy_value.position_input_identity import (
    shogi_position_feature_builder_for_input_module,
    shogi_position_input_identity_for_input_module,
)
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_RICH_POSITION_INPUT_MODULE_ID,
)
from intrep.domains.shogi.engine_analysis import ShogiEngineAnalysis
from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPairRelationEdges,
    ShogiPositionFeatures,
    validate_shogi_rich_position_feature_structure,
)
from intrep.representation.inputs.shogi_position_features.position_schema import (
    PAIR_RELATION_COUNT,
    SHOGI_RICH_POSITION_ELEMENT_COUNT,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
)
from intrep.representation.inputs.shogi_position_features.position_alpha_zero_like import (
    validate_shogi_alpha_zero_like_position_feature_structure,
)
from intrep.representation.inputs.shogi_position_features.position_dlshogi_like import (
    validate_shogi_dlshogi_like_position_feature_structure,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_split_global import (
    validate_shogi_minimal_split_global_position_feature_structure,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    validate_shogi_minimal_single_global_position_feature_structure,
)
from intrep.representation.outputs.shogi_policy_plane_encoding import SHOGI_POLICY_PLANE_ACTION_COUNT

SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA = "intrep.shogi_policy_value_tensor_cache"
SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA = "intrep.shogi_policy_value_tensor_cache_shard"
DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME = "legal-move"
DEFAULT_SHOGI_POLICY_PLANE_VALUE_TENSOR_CACHE_NAME = "policy-plane"
ShogiPolicyValueTensorCacheSample = LegalMovePolicyValueTensorSample | CompactPolicyPlaneValueTensorSample
SHOGI_POLICY_VALUE_TENSOR_CACHE_STORAGE_DTYPES: dict[str, str] = {
    "position_features.global_feature_ids": "uint16",
    "position_features.square_feature_ids": "uint16",
    "position_features.piece_feature_ids": "uint16",
    "position_features.line_feature_ids": "uint16",
    "position_features.pair_relation_edges.source_element_indices": "uint16",
    "position_features.pair_relation_edges.target_element_indices": "uint16",
    "position_features.pair_relation_edges.relation_ids": "uint8",
    "legal_move.legal_move_features": "uint16",
    "legal_move.label": "uint16",
    "legal_move.policy_targets": "float32",
    "legal_move.value_target": "float32",
    "policy_plane.legal_action_indices": "uint16",
    "policy_plane.target_action_indices": "uint16",
    "policy_plane.target_weights": "float32",
    "policy_plane.policy_plane_label": "uint16",
    "policy_plane.value_target": "float32",
}


@dataclass(frozen=True)
class ShogiPolicyValueTensorCache:
    output_space: str
    train_samples: Sequence[ShogiPolicyValueTensorCacheSample]
    eval_samples: Sequence[ShogiPolicyValueTensorCacheSample]
    train_policy_target_summary: dict[str, float | int]
    eval_policy_target_summary: dict[str, float | int]


def default_shogi_policy_value_tensor_cache_path(
    data_selection_path: Path,
    *,
    output_space: str = SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
) -> Path:
    validate_shogi_policy_value_output_space(output_space)
    cache_name = (
        DEFAULT_SHOGI_POLICY_PLANE_VALUE_TENSOR_CACHE_NAME
        if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE
        else DEFAULT_SHOGI_POLICY_VALUE_TENSOR_CACHE_NAME
    )
    return data_selection_path.parent / "cache" / cache_name


def build_shogi_policy_value_tensor_cache(
    *,
    data_selection_path: Path,
    output_path: Path | None = None,
    shard_examples: int = 100_000,
    shard_games: int | None = None,
    resume: bool = False,
    output_space: str = SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
) -> dict[str, object]:
    validate_shogi_policy_value_output_space(output_space)
    shogi_position_input_identity_for_input_module(input_module)
    if shard_games is not None:
        shard_examples = shard_games
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    cache_dir = output_path or default_shogi_policy_value_tensor_cache_path(data_selection_path, output_space=output_space)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    train_summary = _empty_policy_target_summary()
    eval_summary = _empty_policy_target_summary()
    output_stats = _empty_output_space_stats()
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
                output_space=output_space,
                input_module=input_module,
            ):
                shards.append(shard)
                _merge_output_space_stats(output_stats, shard)
                summary = train_summary if split == "train" else eval_summary
                _merge_policy_target_summary(summary, _object_dict(shard["policy_target_summary"]))

    manifest = {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        **_input_feature_identity(input_module),
        "storage_dtypes": SHOGI_POLICY_VALUE_TENSOR_CACHE_STORAGE_DTYPES,
        "output_space": output_space,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection, root=data_selection_path.parent),
        "shard_examples": shard_examples,
        "train_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "train"),
        "eval_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "eval"),
        "skipped_example_count": sum(int(shard.get("skipped_example_count", 0)) for shard in shards),
        **_finalize_output_space_stats(output_space, output_stats),
        "train_policy_target_summary": _finalize_policy_target_summary(train_summary),
        "eval_policy_target_summary": _finalize_policy_target_summary(eval_summary),
        "shards": shards,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        "path": str(cache_dir),
        "output_space": output_space,
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
    output_space: str = SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
) -> dict[str, object]:
    validate_shogi_policy_value_output_space(output_space)
    shogi_position_input_identity_for_input_module(input_module)
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
    source_examples = _source_examples_for_range(
        source,
        data_selection=data_selection,
        analyses_by_position=analyses_by_position,
        start_index=source_example_start_index,
        end_index=source_example_end_index,
    )
    examples = [
        (source_example_start_index + offset, example)
        for offset, example in enumerate(source_examples)
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
        output_space=output_space,
        input_module=input_module,
    )


def write_shogi_policy_value_tensor_cache_manifest(
    *,
    data_selection_path: Path,
    cache_dir: Path,
    shard_examples: int = 100_000,
    shard_games: int | None = None,
    output_space: str | None = None,
    input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
) -> dict[str, object]:
    shogi_position_input_identity_for_input_module(input_module)
    if shard_games is not None:
        shard_examples = shard_games
    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    shards = sorted(
        (_load_shard_manifest_file(path, input_module=input_module) for path in cache_dir.glob("*/*.json")),
        key=lambda shard: (
            str(shard["split"]),
            int(shard["source_index"]),
            int(shard["source_example_start_index"]),
        ),
    )
    if not shards:
        raise ValueError("tensor cache must contain at least one shard manifest")
    shard_output_spaces = {_shard_output_space(shard) for shard in shards}
    if len(shard_output_spaces) != 1:
        raise ValueError("tensor cache shards must have one output_space")
    inferred_output_space = next(iter(shard_output_spaces))
    if output_space is None:
        output_space = inferred_output_space
    validate_shogi_policy_value_output_space(output_space)
    if output_space != inferred_output_space:
        raise ValueError("tensor cache shard output_space does not match requested output_space")
    train_summary = _empty_policy_target_summary()
    eval_summary = _empty_policy_target_summary()
    output_stats = _empty_output_space_stats()
    for shard in shards:
        if shard.get("data_selection_path") != str(data_selection_path):
            raise ValueError(f"tensor cache shard data_selection_path does not match: {shard['path']}")
        _merge_output_space_stats(output_stats, shard)
        summary = train_summary if shard["split"] == "train" else eval_summary
        _merge_policy_target_summary(summary, _object_dict(shard["policy_target_summary"]))

    manifest = {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA,
        **_input_feature_identity(input_module),
        "storage_dtypes": SHOGI_POLICY_VALUE_TENSOR_CACHE_STORAGE_DTYPES,
        "output_space": output_space,
        "data_selection_path": str(data_selection_path),
        "data_selection": shogi_policy_value_data_selection_to_json(data_selection, root=data_selection_path.parent),
        "shard_examples": shard_examples,
        "train_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "train"),
        "eval_count": sum(int(shard["sample_count"]) for shard in shards if shard["split"] == "eval"),
        "skipped_example_count": sum(int(shard.get("skipped_example_count", 0)) for shard in shards),
        **_finalize_output_space_stats(output_space, output_stats),
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
    expected_output_space: str | None = None,
    expected_input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
) -> ShogiPolicyValueTensorCache:
    manifest_path = path / "manifest.json"
    manifest = _object_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache schema")
    _validate_input_feature_identity(manifest, artifact_name="tensor cache", expected_input_module=expected_input_module)
    _validate_storage_dtypes(manifest, artifact_name="tensor cache")
    output_space = _manifest_output_space(manifest)
    if expected_output_space is not None:
        validate_shogi_policy_value_output_space(expected_output_space)
        if output_space != expected_output_space:
            raise ValueError(f"tensor cache output_space={output_space} does not match expected {expected_output_space}")
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
    train_samples, eval_samples = _load_tensor_cache_sequences(
        path,
        manifest,
        output_space,
        train_shards,
        eval_shards,
        input_module=expected_input_module,
    )
    return ShogiPolicyValueTensorCache(
        output_space=output_space,
        train_samples=train_samples,
        eval_samples=eval_samples,
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


def _input_feature_identity(input_module: str) -> dict[str, object]:
    return shogi_position_input_identity_for_input_module(input_module)


def _validate_input_feature_identity(
    payload: dict[str, object],
    *,
    artifact_name: str,
    expected_input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
) -> None:
    expected = shogi_position_input_identity_for_input_module(expected_input_module)
    if payload.get("input_schema_id") != expected["input_schema_id"]:
        raise ValueError(f"unsupported shogi policy/value {artifact_name} input schema")
    if payload.get("input_feature_manifest_hash") != expected["input_feature_manifest_hash"]:
        raise ValueError(f"unsupported shogi policy/value {artifact_name} input feature manifest")
    if payload.get("input_feature_manifest") != expected["input_feature_manifest"]:
        raise ValueError(f"unsupported shogi policy/value {artifact_name} input feature manifest")


def _validate_storage_dtypes(payload: dict[str, object], *, artifact_name: str) -> None:
    if payload.get("storage_dtypes") != SHOGI_POLICY_VALUE_TENSOR_CACHE_STORAGE_DTYPES:
        raise ValueError(f"unsupported shogi policy/value {artifact_name} storage dtypes")


def _manifest_output_space(manifest: dict[str, object]) -> str:
    output_space = manifest.get("output_space", SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE)
    if not isinstance(output_space, str):
        raise ValueError("tensor cache output_space must be a string")
    validate_shogi_policy_value_output_space(output_space)
    return output_space


def _shard_output_space(shard: dict[str, object]) -> str:
    output_space = shard.get("output_space", SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE)
    if not isinstance(output_space, str):
        raise ValueError("tensor cache shard output_space must be a string")
    validate_shogi_policy_value_output_space(output_space)
    return output_space


def _empty_output_space_stats() -> dict[str, int]:
    return {
        "max_legal_move_count": 0,
        "max_legal_action_count": 0,
        "max_target_action_count": 0,
    }


def _merge_output_space_stats(target: dict[str, int], shard: dict[str, object]) -> None:
    for key in ("max_legal_move_count", "max_legal_action_count", "max_target_action_count"):
        if key in shard:
            target[key] = max(target[key], int(shard[key]))


def _finalize_output_space_stats(output_space: str, stats: dict[str, int]) -> dict[str, int]:
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE:
        return {"max_legal_move_count": int(stats["max_legal_move_count"])}
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE:
        return {
            "max_legal_action_count": int(stats["max_legal_action_count"]),
            "max_target_action_count": int(stats["max_target_action_count"]),
        }
    raise ValueError(f"unsupported shogi policy/value output space: {output_space}")


def _split_sources(
    data_selection: ShogiPolicyValueDataSelection,
    split: str,
) -> tuple[ShogiPolicyValueDataSelectionSource, ...]:
    if split == "train":
        return data_selection.train_sources
    if split == "eval":
        return data_selection.eval_sources
    raise ValueError("split must be train or eval")


class ShardedLegalMovePolicyValueTensorSamples(Sequence[LegalMovePolicyValueTensorSample]):
    def __init__(
        self,
        cache_dir: Path,
        shards: Sequence[dict[str, object]],
        *,
        max_legal_move_count: int,
        input_module: str,
    ) -> None:
        self.cache_dir = cache_dir
        self.shards = tuple(shards)
        self.max_legal_move_count = max_legal_move_count
        self.input_module = input_module
        self.offsets: list[int] = []
        self.sequential_access_preferred = True
        offset = 0
        for shard in self.shards:
            self.offsets.append(offset)
            offset += int(shard["sample_count"])
        self.sample_count = offset
        self._loaded_shard_index: int | None = None
        self._loaded_samples: list[LegalMovePolicyValueTensorSample] = []

    def __len__(self) -> int:
        return self.sample_count

    @overload
    def __getitem__(self, index: int) -> LegalMovePolicyValueTensorSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[LegalMovePolicyValueTensorSample]:
        ...

    def __getitem__(self, index: int | slice) -> LegalMovePolicyValueTensorSample | list[LegalMovePolicyValueTensorSample]:
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

    def _load_shard_samples(self, shard_index: int) -> list[LegalMovePolicyValueTensorSample]:
        if self._loaded_shard_index == shard_index:
            return self._loaded_samples
        shard = self.shards[shard_index]
        payload = _load_shard(self.cache_dir / str(shard["path"]), input_module=self.input_module)
        self._loaded_samples = [
            _legal_move_sample_from_payload(item, input_module=self.input_module) for item in payload["samples"]
        ]
        self._loaded_shard_index = shard_index
        return self._loaded_samples


class ShardedCompactPolicyPlaneValueTensorSamples(Sequence[CompactPolicyPlaneValueTensorSample]):
    def __init__(self, cache_dir: Path, shards: Sequence[dict[str, object]], *, input_module: str) -> None:
        self.cache_dir = cache_dir
        self.shards = tuple(shards)
        self.input_module = input_module
        self.offsets: list[int] = []
        self.sequential_access_preferred = True
        offset = 0
        for shard in self.shards:
            self.offsets.append(offset)
            offset += int(shard["sample_count"])
        self.sample_count = offset
        self._loaded_shard_index: int | None = None
        self._loaded_samples: list[CompactPolicyPlaneValueTensorSample] = []

    def __len__(self) -> int:
        return self.sample_count

    @overload
    def __getitem__(self, index: int) -> CompactPolicyPlaneValueTensorSample:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[CompactPolicyPlaneValueTensorSample]:
        ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> CompactPolicyPlaneValueTensorSample | list[CompactPolicyPlaneValueTensorSample]:
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

    def _load_shard_samples(self, shard_index: int) -> list[CompactPolicyPlaneValueTensorSample]:
        if self._loaded_shard_index == shard_index:
            return self._loaded_samples
        shard = self.shards[shard_index]
        payload = _load_shard(self.cache_dir / str(shard["path"]), input_module=self.input_module)
        self._loaded_samples = [
            _compact_policy_plane_sample_from_payload(item, input_module=self.input_module) for item in payload["samples"]
        ]
        self._loaded_shard_index = shard_index
        return self._loaded_samples


def _load_tensor_cache_sequences(
    path: Path,
    manifest: dict[str, object],
    output_space: str,
    train_shards: Sequence[dict[str, object]],
    eval_shards: Sequence[dict[str, object]],
    input_module: str,
) -> tuple[Sequence[ShogiPolicyValueTensorCacheSample], Sequence[ShogiPolicyValueTensorCacheSample]]:
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE:
        max_legal_move_count = int(manifest["max_legal_move_count"])
        return (
            ShardedLegalMovePolicyValueTensorSamples(
                path,
                train_shards,
                max_legal_move_count=max_legal_move_count,
                input_module=input_module,
            ),
            ShardedLegalMovePolicyValueTensorSamples(
                path,
                eval_shards,
                max_legal_move_count=max_legal_move_count,
                input_module=input_module,
            ),
        )
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE:
        return (
            ShardedCompactPolicyPlaneValueTensorSamples(path, train_shards, input_module=input_module),
            ShardedCompactPolicyPlaneValueTensorSamples(path, eval_shards, input_module=input_module),
        )
    raise ValueError(f"unsupported shogi policy/value output space: {output_space}")


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
    output_space: str,
    input_module: str,
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
                    output_space=output_space,
                    input_module=input_module,
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
                output_space=output_space,
                input_module=input_module,
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


def _source_examples_for_range(
    source: ShogiPolicyValueDataSelectionSource,
    *,
    data_selection: ShogiPolicyValueDataSelection,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    start_index: int,
    end_index: int,
) -> list[ShogiMovePolicyValueExample]:
    if source.kind == "shogi_policy_value_examples_jsonl":
        max_examples = source.max_examples
        if max_examples is not None:
            end_index = min(end_index, max_examples)
        if end_index <= start_index:
            return []
        return _load_shogi_move_policy_value_examples_jsonl_range(
            source.path,
            start_index=start_index,
            end_index=end_index,
        )
    return [
        example
        for source_example_index, example in enumerate(
            _source_examples(source, data_selection=data_selection, analyses_by_position=analyses_by_position)
        )
        if start_index <= source_example_index < end_index
    ]


def _load_shogi_move_policy_value_examples_jsonl_range(
    path: str | Path,
    *,
    start_index: int,
    end_index: int,
) -> list[ShogiMovePolicyValueExample]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index <= start_index:
        raise ValueError("end_index must be greater than start_index")
    examples: list[ShogiMovePolicyValueExample] = []
    index = 0
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if index >= end_index:
                break
            if index >= start_index:
                examples.append(shogi_move_policy_value_example_from_json(json.loads(stripped)))
            index += 1
    return examples


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
    output_space: str,
    input_module: str,
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
        output_space=output_space,
        input_module=input_module,
    )
    if resume and shard_path.exists():
        loaded = _try_load_matching_shard(shard_path, expected, input_module=input_module)
        if loaded is not None:
            return loaded

    shard_examples = [example for _source_example_index, example in examples]
    samples = _tensorize_examples_for_output_space(shard_examples, output_space, input_module=input_module)
    summary = _policy_target_summary(shard_examples)
    output_stats = _sample_output_space_stats(samples, output_space)
    payload = {
        **expected,
        "sample_count": len(samples),
        **output_stats,
        "policy_target_summary": summary,
        "skipped_example_count": 0,
        "failures": [],
        "samples": [_sample_to_payload(sample, output_space) for sample in samples],
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
    output_space: str,
    input_module: str,
) -> dict[str, object]:
    return {
        "schema_version": SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA,
        **_input_feature_identity(input_module),
        "storage_dtypes": SHOGI_POLICY_VALUE_TENSOR_CACHE_STORAGE_DTYPES,
        "output_space": output_space,
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


def _try_load_matching_shard(shard_path: Path, expected: dict[str, object], *, input_module: str) -> dict[str, object] | None:
    manifest_path = _shard_manifest_path(shard_path)
    if manifest_path.exists() and shard_path.exists():
        try:
            manifest = _load_shard_manifest_file(manifest_path, input_module=input_module)
        except Exception:  # noqa: BLE001
            return None
        for key, value in expected.items():
            if manifest.get(key) != value:
                return None
        if int(manifest.get("sample_count", 0)) < 0:
            return None
        return manifest

    try:
        payload = _load_shard(shard_path, input_module=input_module)
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


def _load_shard_manifest_file(path: Path, *, input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID) -> dict[str, object]:
    payload = _object_dict(json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache shard manifest schema")
    _validate_input_feature_identity(payload, artifact_name="tensor cache shard", expected_input_module=input_module)
    _validate_storage_dtypes(payload, artifact_name="tensor cache shard")
    return payload


def _write_shard_manifest_file(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _load_shard(path: Path, *, input_module: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != SHOGI_POLICY_VALUE_TENSOR_CACHE_SHARD_SCHEMA:
        raise ValueError("unsupported shogi policy/value tensor cache shard schema")
    _validate_input_feature_identity(payload, artifact_name="tensor cache shard", expected_input_module=input_module)
    _validate_storage_dtypes(payload, artifact_name="tensor cache shard")
    return payload


def _shard_manifest(payload: dict[str, object]) -> dict[str, object]:
    keys = [
        "schema_version",
        "input_schema_id",
        "input_feature_manifest",
        "input_feature_manifest_hash",
        "storage_dtypes",
        "output_space",
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
        "max_legal_move_count",
        "max_legal_action_count",
        "max_target_action_count",
        "policy_target_summary",
        "skipped_example_count",
        "failures",
        "path",
    ]
    return {key: payload[key] for key in keys if key in payload}


def _tensorize_examples_for_output_space(
    examples: Sequence[ShogiMovePolicyValueExample],
    output_space: str,
    *,
    input_module: str,
) -> list[ShogiPolicyValueTensorCacheSample]:
    position_features_from_sfen = shogi_position_feature_builder_for_input_module(input_module)
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE:
        return tensorize_legal_move_policy_value_examples(
            examples,
            position_features_from_sfen=position_features_from_sfen,
        )
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE:
        return tensorize_compact_policy_plane_value_examples(
            examples,
            position_features_from_sfen=position_features_from_sfen,
        )
    raise ValueError(f"unsupported shogi policy/value output space: {output_space}")


def _sample_output_space_stats(
    samples: Sequence[ShogiPolicyValueTensorCacheSample],
    output_space: str,
) -> dict[str, int]:
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE:
        return {
            "max_legal_move_count": max(
                (int(sample.legal_move_features.shape[0]) for sample in samples if isinstance(sample, LegalMovePolicyValueTensorSample)),
                default=0,
            )
        }
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE:
        return {
            "max_legal_action_count": max(
                (
                    int(sample.legal_action_indices.shape[0])
                    for sample in samples
                    if isinstance(sample, CompactPolicyPlaneValueTensorSample)
                ),
                default=0,
            ),
            "max_target_action_count": max(
                (
                    int(sample.target_action_indices.shape[0])
                    for sample in samples
                    if isinstance(sample, CompactPolicyPlaneValueTensorSample)
                ),
                default=0,
            ),
        }
    raise ValueError(f"unsupported shogi policy/value output space: {output_space}")


def _sample_to_payload(sample: ShogiPolicyValueTensorCacheSample, output_space: str) -> dict[str, Any]:
    if output_space == SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE:
        if not isinstance(sample, CompactPolicyPlaneValueTensorSample):
            raise TypeError("policy-plane tensor cache requires compact policy-plane samples")
        return _compact_policy_plane_sample_to_payload(sample)
    if not isinstance(sample, LegalMovePolicyValueTensorSample):
        raise TypeError("legal-move tensor cache requires legal-move samples")
    return _legal_move_sample_to_payload(sample)


def _legal_move_sample_to_payload(sample: LegalMovePolicyValueTensorSample) -> dict[str, Any]:
    return {
        "position_features": _position_features_to_payload(sample.position_features),
        "legal_move_features": _uint16_tensor(
            "legal_move_features",
            sample.legal_move_features,
            maximum_exclusive=SHOGI_RICH_POSITION_ELEMENT_COUNT,
        ),
        "label": _uint16_tensor("label", sample.label),
        "policy_targets": sample.policy_targets.to(dtype=torch.float32),
        "value_target": sample.value_target.to(dtype=torch.float32),
    }


def _legal_move_sample_from_payload(payload: Any, *, input_module: str) -> LegalMovePolicyValueTensorSample:
    if not isinstance(payload, dict):
        raise ValueError("tensor cache sample must be a mapping")
    return LegalMovePolicyValueTensorSample(
        position_features=_position_features_from_payload(payload["position_features"], input_module=input_module),
        legal_move_features=payload["legal_move_features"].to(dtype=torch.long),
        label=payload["label"].to(dtype=torch.long),
        policy_targets=payload["policy_targets"].to(dtype=torch.float32),
        value_target=payload["value_target"].to(dtype=torch.float32),
    )


def _compact_policy_plane_sample_to_payload(sample: CompactPolicyPlaneValueTensorSample) -> dict[str, Any]:
    return {
        "position_features": _position_features_to_payload(sample.position_features),
        "legal_action_indices": _uint16_tensor(
            "legal_action_indices",
            sample.legal_action_indices,
            maximum_exclusive=SHOGI_POLICY_PLANE_ACTION_COUNT,
        ),
        "target_action_indices": _uint16_tensor(
            "target_action_indices",
            sample.target_action_indices,
            maximum_exclusive=SHOGI_POLICY_PLANE_ACTION_COUNT,
        ),
        "target_weights": sample.target_weights.to(dtype=torch.float32),
        "policy_plane_label": _uint16_tensor(
            "policy_plane_label",
            sample.policy_plane_label,
            maximum_exclusive=SHOGI_POLICY_PLANE_ACTION_COUNT,
        ),
        "value_target": sample.value_target.to(dtype=torch.float32),
    }


def _compact_policy_plane_sample_from_payload(payload: Any, *, input_module: str) -> CompactPolicyPlaneValueTensorSample:
    if not isinstance(payload, dict):
        raise ValueError("tensor cache sample must be a mapping")
    return CompactPolicyPlaneValueTensorSample(
        position_features=_position_features_from_payload(payload["position_features"], input_module=input_module),
        legal_action_indices=payload["legal_action_indices"].to(dtype=torch.long),
        target_action_indices=payload["target_action_indices"].to(dtype=torch.long),
        target_weights=payload["target_weights"].to(dtype=torch.float32),
        policy_plane_label=payload["policy_plane_label"].to(dtype=torch.long),
        value_target=payload["value_target"].to(dtype=torch.float32),
    )


def _position_features_to_payload(features: ShogiPositionFeatures) -> dict[str, Any]:
    return {
        "global_feature_ids": _position_feature_tensor("global_feature_ids", features.global_feature_ids),
        "square_feature_ids": _position_feature_tensor("square_feature_ids", features.square_feature_ids),
        "piece_feature_ids": _position_feature_tensor("piece_feature_ids", features.piece_feature_ids),
        "line_feature_ids": _position_feature_tensor("line_feature_ids", features.line_feature_ids),
        "pair_relation_edges": {
            "source_element_indices": _uint16_tensor(
                "pair_relation_edges.source_element_indices",
                features.pair_relation_edges.source_element_indices,
                maximum_exclusive=SHOGI_RICH_POSITION_ELEMENT_COUNT,
            ),
            "target_element_indices": _uint16_tensor(
                "pair_relation_edges.target_element_indices",
                features.pair_relation_edges.target_element_indices,
                maximum_exclusive=SHOGI_RICH_POSITION_ELEMENT_COUNT,
            ),
            "relation_ids": _uint8_tensor(
                "pair_relation_edges.relation_ids",
                features.pair_relation_edges.relation_ids,
                maximum_exclusive=PAIR_RELATION_COUNT,
            ),
        },
    }


def _position_features_from_payload(payload: Any, *, input_module: str) -> ShogiPositionFeatures:
    if not isinstance(payload, dict):
        raise ValueError("position features payload must be a mapping")
    features = ShogiPositionFeatures(
        global_feature_ids=payload["global_feature_ids"].to(dtype=torch.long),
        square_feature_ids=payload["square_feature_ids"].to(dtype=torch.long),
        piece_feature_ids=payload["piece_feature_ids"].to(dtype=torch.long),
        line_feature_ids=payload["line_feature_ids"].to(dtype=torch.long),
        pair_relation_edges=_pair_relation_edges_from_payload(payload["pair_relation_edges"]),
    )
    _validate_position_feature_structure_for_input_module(features, input_module=input_module)
    return features


def _validate_position_feature_structure_for_input_module(
    features: ShogiPositionFeatures,
    *,
    input_module: str,
) -> None:
    if input_module == SHOGI_RICH_POSITION_INPUT_MODULE_ID:
        validate_shogi_rich_position_feature_structure(features)
        return
    if input_module == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        validate_shogi_alpha_zero_like_position_feature_structure(features)
        return
    if input_module == SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID:
        validate_shogi_dlshogi_like_position_feature_structure(features)
        return
    if input_module == SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID:
        validate_shogi_minimal_single_global_position_feature_structure(features)
        return
    if input_module == SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID:
        validate_shogi_minimal_split_global_position_feature_structure(features)
        return
    raise ValueError(f"unsupported shogi position input module: {input_module}")


def _pair_relation_edges_from_payload(payload: Any) -> ShogiPairRelationEdges:
    if not isinstance(payload, dict):
        raise ValueError("pair relation edges payload must be a mapping")
    return ShogiPairRelationEdges(
        source_element_indices=payload["source_element_indices"].to(dtype=torch.long),
        target_element_indices=payload["target_element_indices"].to(dtype=torch.long),
        relation_ids=payload["relation_ids"].to(dtype=torch.long),
    )


def _position_feature_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    return _uint16_tensor(name, tensor, maximum_exclusive=SHOGI_POSITION_FEATURE_VOCAB_SIZE)


def _uint16_tensor(name: str, tensor: torch.Tensor, *, maximum_exclusive: int | None = None) -> torch.Tensor:
    _validate_non_negative_integer_tensor(
        name,
        tensor,
        maximum_exclusive=maximum_exclusive,
        storage_maximum_inclusive=65535,
    )
    return tensor.to(dtype=torch.uint16)


def _uint8_tensor(name: str, tensor: torch.Tensor, *, maximum_exclusive: int | None = None) -> torch.Tensor:
    _validate_non_negative_integer_tensor(
        name,
        tensor,
        maximum_exclusive=maximum_exclusive,
        storage_maximum_inclusive=255,
    )
    return tensor.to(dtype=torch.uint8)


def _validate_non_negative_integer_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    maximum_exclusive: int | None,
    storage_maximum_inclusive: int,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.uint16):
        raise ValueError(f"{name} must use an integer dtype")
    if tensor.numel() == 0:
        return
    minimum = int(tensor.min().item())
    if minimum < 0:
        raise ValueError(f"{name} must be non-negative")
    if maximum_exclusive is not None:
        maximum = int(tensor.max().item())
        if maximum >= maximum_exclusive:
            raise ValueError(f"{name} must be less than {maximum_exclusive}")
    maximum = int(tensor.max().item())
    if maximum > storage_maximum_inclusive:
        raise ValueError(f"{name} must be less than or equal to {storage_maximum_inclusive}")


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
