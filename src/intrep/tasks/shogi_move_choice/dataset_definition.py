from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intrep.tasks.shogi_move_choice.data import load_shogi_move_choice_examples_from_game_records_jsonl
from intrep.tasks.shogi_move_choice.examples import ShogiMoveChoiceExample


@dataclass(frozen=True)
class ShogiMoveChoiceDatasetSource:
    kind: str
    path: Path


@dataclass(frozen=True)
class ShogiMoveChoiceDatasetDefinition:
    name: str
    objective: str
    value_target_source: str
    score_cp_scale: float
    train_sources: tuple[ShogiMoveChoiceDatasetSource, ...]
    eval_sources: tuple[ShogiMoveChoiceDatasetSource, ...]


def load_shogi_move_choice_dataset_definition(path: str | Path) -> ShogiMoveChoiceDatasetDefinition:
    definition_path = Path(path)
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    root = definition_path.parent
    definition = ShogiMoveChoiceDatasetDefinition(
        name=str(payload["name"]),
        objective=str(payload["objective"]),
        value_target_source=str(payload["value_target_source"]),
        score_cp_scale=float(payload["score_cp_scale"]),
        train_sources=_sources_from_json(payload.get("train_sources"), root=root),
        eval_sources=_sources_from_json(payload.get("eval_sources"), root=root),
    )
    _validate_value_target_source(definition)
    _validate_split(definition)
    return definition


def load_shogi_move_choice_dataset_examples(
    definition: ShogiMoveChoiceDatasetDefinition,
) -> tuple[list[ShogiMoveChoiceExample], list[ShogiMoveChoiceExample]]:
    train_examples = _load_sources(
        definition.train_sources,
        value_target_source=definition.value_target_source,
        score_cp_scale=definition.score_cp_scale,
    )
    eval_examples = _load_sources(
        definition.eval_sources,
        value_target_source=definition.value_target_source,
        score_cp_scale=definition.score_cp_scale,
    )
    return train_examples, eval_examples


def shogi_move_choice_dataset_definition_to_json(definition: ShogiMoveChoiceDatasetDefinition) -> dict[str, Any]:
    return {
        "name": definition.name,
        "objective": definition.objective,
        "value_target_source": definition.value_target_source,
        "score_cp_scale": definition.score_cp_scale,
        "train_sources": [_source_to_json(source) for source in definition.train_sources],
        "eval_sources": [_source_to_json(source) for source in definition.eval_sources],
    }


def _sources_from_json(value: object, *, root: Path) -> tuple[ShogiMoveChoiceDatasetSource, ...]:
    if not isinstance(value, list):
        raise ValueError("dataset sources must be a list")
    if not value:
        raise ValueError("dataset sources must be a non-empty list")
    sources: list[ShogiMoveChoiceDatasetSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("dataset source must be an object")
        kind = str(item["kind"])
        if kind != "game_records_jsonl":
            raise ValueError("dataset source kind must be game_records_jsonl")
        source_path = Path(str(item["path"]))
        if not source_path.is_absolute():
            source_path = root / source_path
        sources.append(ShogiMoveChoiceDatasetSource(kind=kind, path=source_path))
    return tuple(sources)


def _validate_split(definition: ShogiMoveChoiceDatasetDefinition) -> None:
    train_sources = {_source_key(source) for source in definition.train_sources}
    eval_sources = {_source_key(source) for source in definition.eval_sources}
    overlap = train_sources & eval_sources
    if overlap:
        raise ValueError("train and eval dataset sources must be split")


def _validate_value_target_source(definition: ShogiMoveChoiceDatasetDefinition) -> None:
    if definition.value_target_source not in {"winner", "yaneuraou_best_score"}:
        raise ValueError("value_target_source must be winner or yaneuraou_best_score")
    if definition.score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")


def _load_sources(
    sources: tuple[ShogiMoveChoiceDatasetSource, ...],
    *,
    value_target_source: str,
    score_cp_scale: float,
) -> list[ShogiMoveChoiceExample]:
    examples: list[ShogiMoveChoiceExample] = []
    for source in sources:
        if source.kind == "game_records_jsonl":
            examples.extend(
                load_shogi_move_choice_examples_from_game_records_jsonl(
                    source.path,
                    value_target_source=value_target_source,
                    score_cp_scale=score_cp_scale,
                )
            )
        else:
            raise ValueError(f"unsupported dataset source kind: {source.kind}")
    if not examples:
        raise ValueError("dataset definition must load at least one example")
    return examples


def _source_to_json(source: ShogiMoveChoiceDatasetSource) -> dict[str, str]:
    return {
        "kind": source.kind,
        "path": str(source.path),
    }


def _source_key(source: ShogiMoveChoiceDatasetSource) -> tuple[str, Path]:
    return source.kind, source.path.resolve()
