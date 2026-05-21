from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from intrep.problems.shogi_policy_value.data import (
    load_shogi_engine_analysis_by_position_jsonl,
    load_shogi_move_policy_value_examples_from_game_records_jsonl_with_engine_analysis,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiMovePolicyValueExample,
    load_shogi_move_policy_value_examples_jsonl,
)
from intrep.domains.shogi.engine_analysis import ShogiEngineAnalysis


@dataclass(frozen=True)
class ShogiPolicyValueDataSelectionSource:
    kind: str
    path: Path
    max_games: int | None = None
    max_examples: int | None = None


@dataclass(frozen=True)
class ShogiPolicyValueTargetConstruction:
    policy: str
    value: str
    policy_temperature_cp: float
    policy_mate_cp: float
    score_cp_scale: float


@dataclass(frozen=True)
class ShogiPolicyValueDataSelection:
    name: str
    objective: str
    target_construction: ShogiPolicyValueTargetConstruction | None
    analysis_sources: tuple[ShogiPolicyValueDataSelectionSource, ...]
    train_sources: tuple[ShogiPolicyValueDataSelectionSource, ...]
    eval_sources: tuple[ShogiPolicyValueDataSelectionSource, ...]


def load_shogi_policy_value_data_selection(path: str | Path) -> ShogiPolicyValueDataSelection:
    selection_path = Path(path)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    root = selection_path.parent
    selection = ShogiPolicyValueDataSelection(
        name=str(payload["name"]),
        objective=str(payload["objective"]),
        target_construction=_target_construction_from_json(payload.get("target_construction")),
        analysis_sources=_sources_from_json(payload.get("analysis_sources", []), root=root, allowed_kinds={"shogi_engine_analysis_jsonl"}, require_non_empty=False),
        train_sources=_sources_from_json(payload.get("train_sources"), root=root),
        eval_sources=_sources_from_json(payload.get("eval_sources"), root=root),
    )
    _validate_target_construction(selection)
    _validate_split(selection)
    return selection


def load_shogi_policy_value_data_selection_examples(
    selection: ShogiPolicyValueDataSelection,
) -> tuple[list[ShogiMovePolicyValueExample], list[ShogiMovePolicyValueExample]]:
    analyses_by_position = load_shogi_engine_analysis_by_position_jsonl(
        tuple(source.path for source in selection.analysis_sources)
    )
    train_examples = _load_sources(selection.train_sources, selection=selection, analyses_by_position=analyses_by_position)
    eval_examples = _load_sources(selection.eval_sources, selection=selection, analyses_by_position=analyses_by_position)
    return train_examples, eval_examples


def shogi_policy_value_data_selection_to_json(selection: ShogiPolicyValueDataSelection, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "name": selection.name,
        "objective": selection.objective,
        **(
            {"target_construction": _target_construction_to_json(selection.target_construction)}
            if selection.target_construction is not None
            else {}
        ),
        **(
            {"analysis_sources": [_source_to_json(source, root=root) for source in selection.analysis_sources]}
            if selection.analysis_sources
            else {}
        ),
        "train_sources": [_source_to_json(source, root=root) for source in selection.train_sources],
        "eval_sources": [_source_to_json(source, root=root) for source in selection.eval_sources],
    }


def _target_construction_from_json(value: object) -> ShogiPolicyValueTargetConstruction | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("target_construction must be an object")
    return ShogiPolicyValueTargetConstruction(
        policy=str(value["policy"]),
        value=str(value["value"]),
        policy_temperature_cp=float(value["policy_temperature_cp"]),
        policy_mate_cp=float(value["policy_mate_cp"]),
        score_cp_scale=float(value["score_cp_scale"]),
    )


def _target_construction_to_json(construction: ShogiPolicyValueTargetConstruction) -> dict[str, str | float]:
    return {
        "policy": construction.policy,
        "policy_temperature_cp": construction.policy_temperature_cp,
        "policy_mate_cp": construction.policy_mate_cp,
        "value": construction.value,
        "score_cp_scale": construction.score_cp_scale,
    }


def _sources_from_json(
    value: object,
    *,
    root: Path,
    allowed_kinds: set[str] | None = None,
    require_non_empty: bool = True,
) -> tuple[ShogiPolicyValueDataSelectionSource, ...]:
    if not isinstance(value, list):
        raise ValueError("data selection sources must be a list")
    if require_non_empty and not value:
        raise ValueError("data selection sources must be a non-empty list")
    sources: list[ShogiPolicyValueDataSelectionSource] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("data selection source must be an object")
        kind = str(item["kind"])
        if allowed_kinds is None:
            allowed_kinds = {"shogi_policy_value_examples_jsonl", "game_records_jsonl"}
        if kind not in allowed_kinds:
            raise ValueError(f"data selection source kind must be one of {sorted(allowed_kinds)}")
        source_path = Path(str(item["path"]))
        if not source_path.is_absolute():
            source_path = root / source_path
        max_games = None
        if "max_games" in item:
            max_games = int(item["max_games"])
            if max_games <= 0:
                raise ValueError("data selection source max_games must be positive")
        max_examples = None
        if "max_examples" in item:
            max_examples = int(item["max_examples"])
            if max_examples <= 0:
                raise ValueError("data selection source max_examples must be positive")
        sources.append(
            ShogiPolicyValueDataSelectionSource(
                kind=kind,
                path=source_path,
                max_games=max_games,
                max_examples=max_examples,
            )
        )
    return tuple(sources)


def _validate_split(selection: ShogiPolicyValueDataSelection) -> None:
    train_sources = {_source_key(source) for source in selection.train_sources}
    eval_sources = {_source_key(source) for source in selection.eval_sources}
    overlap = train_sources & eval_sources
    if overlap:
        raise ValueError("train and eval data selection sources must be split")


def _validate_target_construction(selection: ShogiPolicyValueDataSelection) -> None:
    construction = selection.target_construction
    if construction is None:
        if any(source.kind == "game_records_jsonl" for source in (*selection.train_sources, *selection.eval_sources)):
            raise ValueError("target_construction is required for game_records_jsonl sources")
        return
    if construction.policy not in {"chosen_move", "decision_usi_multipv", "engine_analysis_multipv", "mcts_visit_counts"}:
        raise ValueError(
            "target_construction.policy must be chosen_move, decision_usi_multipv, engine_analysis_multipv, or mcts_visit_counts"
        )
    if construction.value not in {"winner", "decision_usi_score", "engine_analysis_score"}:
        raise ValueError("target_construction.value must be winner, decision_usi_score, or engine_analysis_score")
    if (
        construction.policy == "engine_analysis_multipv" or construction.value == "engine_analysis_score"
    ) and not selection.analysis_sources:
        raise ValueError("analysis_sources must be non-empty when target_construction uses engine analysis")
    if construction.score_cp_scale <= 0:
        raise ValueError("score_cp_scale must be positive")
    if construction.policy_temperature_cp <= 0:
        raise ValueError("policy_temperature_cp must be positive")
    if construction.policy_mate_cp <= 0:
        raise ValueError("policy_mate_cp must be positive")


def _load_sources(
    sources: tuple[ShogiPolicyValueDataSelectionSource, ...],
    *,
    selection: ShogiPolicyValueDataSelection,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
) -> list[ShogiMovePolicyValueExample]:
    examples: list[ShogiMovePolicyValueExample] = []
    for source in sources:
        if source.kind == "shogi_policy_value_examples_jsonl":
            examples.extend(load_shogi_move_policy_value_examples_jsonl(source.path, max_examples=source.max_examples))
        elif source.kind == "game_records_jsonl":
            if selection.target_construction is None:
                raise ValueError("target_construction is required for game_records_jsonl sources")
            examples.extend(
                load_shogi_move_policy_value_examples_from_game_records_jsonl_with_engine_analysis(
                    source.path,
                    policy_target_construction=selection.target_construction.policy,
                    value_target_construction=selection.target_construction.value,
                    analyses_by_position=analyses_by_position,
                    policy_temperature_cp=selection.target_construction.policy_temperature_cp,
                    policy_mate_cp=selection.target_construction.policy_mate_cp,
                    score_cp_scale=selection.target_construction.score_cp_scale,
                    max_games=source.max_games,
                )
            )
        else:
            raise ValueError(f"unsupported data selection source kind: {source.kind}")
    if not examples:
        raise ValueError("data selection must load at least one example")
    return examples


def _source_to_json(source: ShogiPolicyValueDataSelectionSource, *, root: Path | None = None) -> dict[str, str | int]:
    path = source.path
    if root is not None:
        try:
            path = path.relative_to(root)
        except ValueError:
            path = source.path
    payload: dict[str, str | int] = {
        "kind": source.kind,
        "path": str(path),
    }
    if source.max_games is not None:
        payload["max_games"] = source.max_games
    if source.max_examples is not None:
        payload["max_examples"] = source.max_examples
    return payload


def _source_key(source: ShogiPolicyValueDataSelectionSource) -> tuple[str, Path, int | None, int | None]:
    return source.kind, source.path.resolve(), source.max_games, source.max_examples
