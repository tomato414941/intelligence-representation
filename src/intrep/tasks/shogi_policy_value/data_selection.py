from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intrep.tasks.shogi_policy_value.data import load_shogi_policy_value_examples_from_game_records_jsonl
from intrep.tasks.shogi_policy_value.examples import ShogiPolicyValueExample


@dataclass(frozen=True)
class ShogiPolicyValueDataSelectionSource:
    kind: str
    path: Path
    max_games: int | None = None
    policy_target_source: str | None = None
    value_target_source: str | None = None


@dataclass(frozen=True)
class ShogiPolicyValueDataSelection:
    # Target policy is not conceptually Data Selection. It lives here for now
    # so one small file can define the selected sources and their target
    # derivation policy until a concrete run needs separate reuse.
    name: str
    objective: str
    policy_target_source: str
    policy_temperature_cp: float
    policy_mate_cp: float
    value_target_source: str
    score_cp_scale: float
    train_sources: tuple[ShogiPolicyValueDataSelectionSource, ...]
    eval_sources: tuple[ShogiPolicyValueDataSelectionSource, ...]


def load_shogi_policy_value_data_selection(path: str | Path) -> ShogiPolicyValueDataSelection:
    selection_path = Path(path)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    root = selection_path.parent
    selection = ShogiPolicyValueDataSelection(
        name=str(payload["name"]),
        objective=str(payload["objective"]),
        policy_target_source=str(payload["policy_target_source"]),
        policy_temperature_cp=float(payload["policy_temperature_cp"]),
        policy_mate_cp=float(payload["policy_mate_cp"]),
        value_target_source=str(payload["value_target_source"]),
        score_cp_scale=float(payload["score_cp_scale"]),
        train_sources=_sources_from_json(payload.get("train_sources"), root=root),
        eval_sources=_sources_from_json(payload.get("eval_sources"), root=root),
    )
    _validate_policy_target_source(selection)
    _validate_value_target_source(selection)
    _validate_split(selection)
    return selection


def load_shogi_policy_value_data_selection_examples(
    selection: ShogiPolicyValueDataSelection,
) -> tuple[list[ShogiPolicyValueExample], list[ShogiPolicyValueExample]]:
    train_examples = _load_sources(
        selection.train_sources,
        policy_target_source=selection.policy_target_source,
        value_target_source=selection.value_target_source,
        policy_temperature_cp=selection.policy_temperature_cp,
        policy_mate_cp=selection.policy_mate_cp,
        score_cp_scale=selection.score_cp_scale,
    )
    eval_examples = _load_sources(
        selection.eval_sources,
        policy_target_source=selection.policy_target_source,
        value_target_source=selection.value_target_source,
        policy_temperature_cp=selection.policy_temperature_cp,
        policy_mate_cp=selection.policy_mate_cp,
        score_cp_scale=selection.score_cp_scale,
    )
    return train_examples, eval_examples


def shogi_policy_value_data_selection_to_json(selection: ShogiPolicyValueDataSelection) -> dict[str, Any]:
    return {
        "name": selection.name,
        "objective": selection.objective,
        "policy_target_source": selection.policy_target_source,
        "policy_temperature_cp": selection.policy_temperature_cp,
        "policy_mate_cp": selection.policy_mate_cp,
        "value_target_source": selection.value_target_source,
        "score_cp_scale": selection.score_cp_scale,
        "train_sources": [_source_to_json(source) for source in selection.train_sources],
        "eval_sources": [_source_to_json(source) for source in selection.eval_sources],
    }


def _sources_from_json(value: object, *, root: Path) -> tuple[ShogiPolicyValueDataSelectionSource, ...]:
    if not isinstance(value, list):
        raise ValueError("data selection sources must be a list")
    if not value:
        raise ValueError("data selection sources must be a non-empty list")
    sources: list[ShogiPolicyValueDataSelectionSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("data selection source must be an object")
        kind = str(item["kind"])
        if kind != "game_records_jsonl":
            raise ValueError("data selection source kind must be game_records_jsonl")
        source_path = Path(str(item["path"]))
        if not source_path.is_absolute():
            source_path = root / source_path
        max_games = None
        if "max_games" in item:
            max_games = int(item["max_games"])
            if max_games <= 0:
                raise ValueError("data selection source max_games must be positive")
        policy_target_source = None
        if "policy_target_source" in item:
            policy_target_source = str(item["policy_target_source"])
            _validate_policy_target_source_value(policy_target_source)
        value_target_source = None
        if "value_target_source" in item:
            value_target_source = str(item["value_target_source"])
            _validate_value_target_source_value(value_target_source)
        sources.append(
            ShogiPolicyValueDataSelectionSource(
                kind=kind,
                path=source_path,
                max_games=max_games,
                policy_target_source=policy_target_source,
                value_target_source=value_target_source,
            )
        )
    return tuple(sources)


def _validate_split(selection: ShogiPolicyValueDataSelection) -> None:
    train_sources = {_source_key(source) for source in selection.train_sources}
    eval_sources = {_source_key(source) for source in selection.eval_sources}
    overlap = train_sources & eval_sources
    if overlap:
        raise ValueError("train and eval data selection sources must be split")


def _validate_value_target_source(selection: ShogiPolicyValueDataSelection) -> None:
    _validate_value_target_source_value(selection.value_target_source)
    if selection.score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")


def _validate_policy_target_source(selection: ShogiPolicyValueDataSelection) -> None:
    _validate_policy_target_source_value(selection.policy_target_source)
    if selection.policy_temperature_cp <= 0:
        raise ValueError("policy_temperature_cp must be positive")
    if selection.policy_mate_cp <= 0:
        raise ValueError("policy_mate_cp must be positive")


def _validate_value_target_source_value(value: str) -> None:
    if value not in {"winner", "yaneuraou_best_score"}:
        raise ValueError("value_target_source must be winner or yaneuraou_best_score")


def _validate_policy_target_source_value(value: str) -> None:
    if value not in {"chosen_move", "usi_multipv"}:
        raise ValueError("policy_target_source must be chosen_move or usi_multipv")


def _load_sources(
    sources: tuple[ShogiPolicyValueDataSelectionSource, ...],
    *,
    policy_target_source: str,
    value_target_source: str,
    policy_temperature_cp: float,
    policy_mate_cp: float,
    score_cp_scale: float,
) -> list[ShogiPolicyValueExample]:
    examples: list[ShogiPolicyValueExample] = []
    for source in sources:
        if source.kind == "game_records_jsonl":
            examples.extend(
                load_shogi_policy_value_examples_from_game_records_jsonl(
                    source.path,
                    policy_target_source=source.policy_target_source or policy_target_source,
                    value_target_source=source.value_target_source or value_target_source,
                    policy_temperature_cp=policy_temperature_cp,
                    policy_mate_cp=policy_mate_cp,
                    score_cp_scale=score_cp_scale,
                    max_games=source.max_games,
                )
            )
        else:
            raise ValueError(f"unsupported data selection source kind: {source.kind}")
    if not examples:
        raise ValueError("data selection must load at least one example")
    return examples


def _source_to_json(source: ShogiPolicyValueDataSelectionSource) -> dict[str, str | int]:
    payload: dict[str, str | int] = {
        "kind": source.kind,
        "path": str(source.path),
    }
    if source.max_games is not None:
        payload["max_games"] = source.max_games
    if source.policy_target_source is not None:
        payload["policy_target_source"] = source.policy_target_source
    if source.value_target_source is not None:
        payload["value_target_source"] = source.value_target_source
    return payload


def _source_key(source: ShogiPolicyValueDataSelectionSource) -> tuple[str, Path, int | None]:
    return source.kind, source.path.resolve(), source.max_games
