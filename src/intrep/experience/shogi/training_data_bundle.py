from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from intrep.problems.shogi_policy_value.data import (
    load_shogi_engine_analysis_by_position_jsonl,
    shogi_move_policy_value_examples_from_game_record,
)
from intrep.problems.shogi_policy_value.examples import (
    ShogiMovePolicyValueExample,
    write_shogi_move_policy_value_examples_jsonl,
)
from intrep.worlds.shogi.engine_analysis import ShogiEngineAnalysis, load_shogi_engine_analysis_jsonl
from intrep.experience.shogi.experience_stats import (
    shogi_actor_pair,
    shogi_actor_pair_counts,
    shogi_checkpoint_actor_summaries,
    shogi_position_stats,
    shogi_train_eval_position_stats,
)
from intrep.worlds.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl
from intrep.worlds.shogi.game_trace import trace_shogi_game_record

ShogiEvalPositionPolicy = Literal["allow_overlap", "exclude_train_position_games"]


def create_shogi_training_data_bundle(
    *,
    train_games: Path | tuple[Path, ...],
    eval_games: Path,
    name: str,
    output_root: Path,
    max_train_games: int | None = None,
    max_eval_games: int | None = None,
    actor_pair_ratios: dict[str, float] | None = None,
    seed: int = 7,
    policy_target_construction: str = "chosen_move",
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    value_target_construction: str = "winner",
    score_cp_scale: float = 600.0,
    analysis_sources: tuple[Path, ...] = (),
    eval_position_policy: ShogiEvalPositionPolicy = "allow_overlap",
    include_position_stats: bool = True,
) -> dict[str, object]:
    output_dir = output_root / name
    train_jsonl = output_dir / "train-examples.jsonl"
    eval_jsonl = output_dir / "eval-examples.jsonl"
    data_selection_json = output_dir / "data-selection.json"
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists():
        raise FileExistsError(f"training data bundle already exists: {output_dir}")
    _validate_max_games(max_train_games, label="max_train_games")
    _validate_max_games(max_eval_games, label="max_eval_games")
    _validate_analysis_sources_for_targets(
        analysis_sources=analysis_sources,
        policy_target_construction=policy_target_construction,
        value_target_construction=value_target_construction,
    )
    _validate_eval_position_policy(eval_position_policy)

    train_paths = _normalize_train_games(train_games)
    available_train_records = _load_records(train_paths)
    available_eval_records = list(iter_shogi_game_records_jsonl(eval_games))
    if not available_train_records:
        raise ValueError("train games must not be empty")
    if not available_eval_records:
        raise ValueError("eval games must not be empty")

    train_records = select_shogi_game_records(
        available_train_records,
        max_games=max_train_games,
        actor_pair_ratios=actor_pair_ratios or {},
        seed=seed,
    )
    selected_eval_records = _limit_records(available_eval_records, max_eval_games)
    eval_records, skipped_eval_games = _apply_eval_position_policy(
        selected_eval_records,
        train_records=train_records,
        eval_position_policy=eval_position_policy,
    )
    if not train_records:
        raise ValueError("training data bundle selection must produce at least one train game")
    if not eval_records:
        raise ValueError("training data bundle selection must produce at least one eval game")
    records = train_records + eval_records
    train_eval_position_stats = (
        shogi_train_eval_position_stats(train_records, eval_records).to_dict()
        if include_position_stats
        else _empty_position_stats()
    )

    output_dir.mkdir(parents=True)
    analysis_jsonls = _copy_analysis_sources(analysis_sources, output_dir=output_dir)
    analyses_by_position = load_shogi_engine_analysis_by_position_jsonl(tuple(analysis_jsonls))
    train_examples = _examples_from_records(
        train_records,
        policy_target_construction=policy_target_construction,
        value_target_construction=value_target_construction,
        analyses_by_position=analyses_by_position,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
        score_cp_scale=score_cp_scale,
    )
    eval_examples = _examples_from_records(
        eval_records,
        policy_target_construction=policy_target_construction,
        value_target_construction=value_target_construction,
        analyses_by_position=analyses_by_position,
        policy_temperature_cp=policy_temperature_cp,
        policy_mate_cp=policy_mate_cp,
        score_cp_scale=score_cp_scale,
    )
    write_shogi_move_policy_value_examples_jsonl(train_jsonl, train_examples)
    write_shogi_move_policy_value_examples_jsonl(eval_jsonl, eval_examples)
    analysis_coverage = shogi_analysis_coverage(train_records, eval_records, analysis_jsonls) if analysis_jsonls else {}

    data_selection = {
        "name": name,
        "objective": "shogi move-choice policy/value",
        "train_sources": [_source_json(train_jsonl.name)],
        "eval_sources": [_source_json(eval_jsonl.name)],
    }
    data_selection_json.write_text(json.dumps(data_selection, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "intrep.shogi_training_data_bundle.v1",
        "example_schema": "shogi_policy_value_example_jsonl",
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "train_source_games_jsonl": [str(path) for path in train_paths],
        "eval_source_games_jsonl": str(eval_games),
        "analysis_source_jsonl": [str(path) for path in analysis_sources],
        "seed": seed,
        "max_train_games": max_train_games,
        "max_eval_games": max_eval_games,
        "eval_position_policy": eval_position_policy,
        "actor_pair_ratios": dict(sorted((actor_pair_ratios or {}).items())),
        "available_train_games": len(available_train_records),
        "available_eval_games": len(available_eval_records),
        "selected_eval_games_before_position_policy": len(selected_eval_records),
        "skipped_eval_games_for_train_position_overlap": skipped_eval_games,
        "game_count": len(records),
        "transition_count": sum(len(record.moves) for record in records),
        "position_stats_included": include_position_stats,
        "position_stats": shogi_position_stats(records).to_dict() if include_position_stats else None,
        **train_eval_position_stats,
        "analysis_coverage": analysis_coverage,
        "actor_pair_counts": shogi_actor_pair_counts(records),
        "train_actor_pair_counts": shogi_actor_pair_counts(train_records),
        "eval_actor_pair_counts": shogi_actor_pair_counts(eval_records),
        "checkpoint_actor_summaries": shogi_checkpoint_actor_summaries(records),
        "train_checkpoint_actor_summaries": shogi_checkpoint_actor_summaries(train_records),
        "eval_checkpoint_actor_summaries": shogi_checkpoint_actor_summaries(eval_records),
        "train_games": len(train_records),
        "eval_games": len(eval_records),
        "target_construction": {
            "policy": policy_target_construction,
            "policy_temperature_cp": policy_temperature_cp,
            "policy_mate_cp": policy_mate_cp,
            "value": value_target_construction,
            "score_cp_scale": score_cp_scale,
        },
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "files": {
            "train": train_jsonl.name,
            "eval": eval_jsonl.name,
            "analysis": [path.name for path in analysis_jsonls],
            "data_selection": data_selection_json.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "training_data_bundle": str(output_dir),
        "data_selection_json": str(data_selection_json),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "analysis_jsonl": [str(path) for path in analysis_jsonls],
        "manifest": str(manifest_path),
        "game_count": len(records),
        "train_games": len(train_records),
        "eval_games": len(eval_records),
    }


def select_shogi_game_records(
    records: list[ShogiGameRecord],
    *,
    max_games: int | None,
    actor_pair_ratios: dict[str, float],
    seed: int,
) -> list[ShogiGameRecord]:
    limit = len(records) if max_games is None else min(max_games, len(records))
    if limit <= 0:
        return []
    groups = _records_by_actor_pair(records)
    if not actor_pair_ratios:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        return shuffled[:limit]

    selected: list[ShogiGameRecord] = []
    for actor_pair, count in _actor_pair_target_counts(groups, actor_pair_ratios, limit).items():
        group = list(groups.get(actor_pair, ()))
        random.Random(f"{seed}:{actor_pair}").shuffle(group)
        selected.extend(group[:count])
    random.Random(f"{seed}:selected").shuffle(selected)
    return selected


def parse_shogi_actor_pair_ratios(values: list[str]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for value in values:
        actor_pair, separator, weight = value.partition("=")
        if not separator or not actor_pair:
            raise ValueError("actor pair ratio must use ACTOR_PAIR=WEIGHT")
        ratios[actor_pair] = float(weight)
    return ratios


def shogi_analysis_coverage(
    train_records: list[ShogiGameRecord],
    eval_records: list[ShogiGameRecord],
    analysis_paths: list[Path],
) -> dict[str, dict[str, float | int]]:
    analyzed_positions = _analyzed_positions(analysis_paths)
    return {
        "train": _position_coverage(train_records, analyzed_positions),
        "eval": _position_coverage(eval_records, analyzed_positions),
    }


def _normalize_train_games(train_games: Path | tuple[Path, ...]) -> tuple[Path, ...]:
    if isinstance(train_games, Path):
        return (train_games,)
    if not train_games:
        raise ValueError("train_games must not be empty")
    return train_games


def _load_records(paths: tuple[Path, ...]) -> list[ShogiGameRecord]:
    records: list[ShogiGameRecord] = []
    for path in paths:
        records.extend(iter_shogi_game_records_jsonl(path))
    return records


def _apply_eval_position_policy(
    records: list[ShogiGameRecord],
    *,
    train_records: list[ShogiGameRecord],
    eval_position_policy: ShogiEvalPositionPolicy,
) -> tuple[list[ShogiGameRecord], int]:
    if eval_position_policy == "allow_overlap":
        return records, 0
    train_positions = _position_set(train_records)
    selected: list[ShogiGameRecord] = []
    skipped = 0
    for record in records:
        if _position_set([record]) & train_positions:
            skipped += 1
            continue
        selected.append(record)
    return selected, skipped


def _actor_pair_target_counts(
    groups: dict[str, list[ShogiGameRecord]],
    ratios: dict[str, float],
    limit: int,
) -> dict[str, int]:
    total_ratio = sum(ratios.values())
    if total_ratio <= 0.0:
        raise ValueError("actor pair ratios must have positive sum")

    targets: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for actor_pair, ratio in sorted(ratios.items()):
        if ratio <= 0.0:
            raise ValueError("actor pair ratios must be positive")
        exact = limit * ratio / total_ratio
        count = min(math.floor(exact), len(groups.get(actor_pair, ())))
        targets[actor_pair] = count
        remainders.append((exact - math.floor(exact), actor_pair))

    selected_count = sum(targets.values())
    for _remainder, actor_pair in sorted(remainders, reverse=True):
        if selected_count >= limit:
            break
        available = len(groups.get(actor_pair, ()))
        if targets[actor_pair] < available:
            targets[actor_pair] += 1
            selected_count += 1
    return {actor_pair: count for actor_pair, count in targets.items() if count > 0}


def _records_by_actor_pair(records: list[ShogiGameRecord]) -> dict[str, list[ShogiGameRecord]]:
    groups: dict[str, list[ShogiGameRecord]] = {}
    for record in records:
        groups.setdefault(shogi_actor_pair(record), []).append(record)
    return groups


def _limit_records(records: list[ShogiGameRecord], max_games: int | None) -> list[ShogiGameRecord]:
    if max_games is None:
        return records
    return records[:max_games]


def _position_set(records: list[ShogiGameRecord]) -> set[str]:
    return {transition.position_sfen for record in records for transition in trace_shogi_game_record(record).transitions}


def _validate_eval_position_policy(eval_position_policy: str) -> None:
    if eval_position_policy not in {"allow_overlap", "exclude_train_position_games"}:
        raise ValueError("eval_position_policy must be allow_overlap or exclude_train_position_games")


def _source_json(path: str) -> dict[str, str]:
    return {"kind": "shogi_policy_value_examples_jsonl", "path": path}


def _examples_from_records(
    records: list[ShogiGameRecord],
    *,
    policy_target_construction: str,
    value_target_construction: str,
    analyses_by_position: dict[str, ShogiEngineAnalysis],
    policy_temperature_cp: float,
    policy_mate_cp: float,
    score_cp_scale: float,
) -> list[ShogiMovePolicyValueExample]:
    examples: list[ShogiMovePolicyValueExample] = []
    for game_index, record in enumerate(records):
        game_examples = shogi_move_policy_value_examples_from_game_record(
            record,
            policy_target_construction=policy_target_construction,
            value_target_construction=value_target_construction,
            analyses_by_position=analyses_by_position,
            policy_temperature_cp=policy_temperature_cp,
            policy_mate_cp=policy_mate_cp,
            score_cp_scale=score_cp_scale,
        )
        for ply_index, example in enumerate(game_examples):
            examples.append(replace(example, game_index=game_index, ply_index=ply_index))
    return examples


def _copy_analysis_sources(paths: tuple[Path, ...], *, output_dir: Path) -> list[Path]:
    output_paths: list[Path] = []
    for index, source in enumerate(paths, start=1):
        name = "analysis.jsonl" if len(paths) == 1 else f"analysis-{index:04d}.jsonl"
        output_path = output_dir / name
        shutil.copyfile(source, output_path)
        output_paths.append(output_path)
    return output_paths


def _analyzed_positions(paths: list[Path]) -> set[str]:
    positions: set[str] = set()
    for path in paths:
        for analysis in load_shogi_engine_analysis_jsonl(path):
            positions.add(analysis.position_sfen)
    return positions


def _position_coverage(records: list[ShogiGameRecord], analyzed_positions: set[str]) -> dict[str, float | int]:
    positions = {
        transition.position_sfen
        for record in records
        for transition in trace_shogi_game_record(record).transitions
    }
    covered = len(positions & analyzed_positions)
    total = len(positions)
    return {
        "positions": total,
        "covered": covered,
        "ratio": covered / total if total else 0.0,
    }


def _empty_position_stats() -> dict[str, object]:
    return {
        "train_position_stats": None,
        "eval_position_stats": None,
        "train_eval_position_overlap_count": None,
        "train_eval_position_overlap_ratio": None,
    }


def _validate_analysis_sources_for_targets(
    *,
    analysis_sources: tuple[Path, ...],
    policy_target_construction: str,
    value_target_construction: str,
) -> None:
    if (
        policy_target_construction == "engine_analysis_multipv"
        or value_target_construction == "engine_analysis_score"
    ) and not analysis_sources:
        raise ValueError("analysis_sources must be non-empty when target construction uses engine analysis")


def _validate_max_games(value: int | None, *, label: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{label} must be positive")
