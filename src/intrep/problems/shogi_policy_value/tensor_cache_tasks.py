from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelectionSource,
    load_shogi_policy_value_data_selection,
)

ShogiTensorCacheSplit = Literal["all", "train", "eval"]
ShogiTensorCacheTask = dict[str, int | str]


def build_shogi_policy_value_tensor_cache_tasks(
    *,
    data_selection_path: Path,
    shard_examples: int,
    split: ShogiTensorCacheSplit = "all",
) -> list[ShogiTensorCacheTask]:
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    if split not in {"all", "train", "eval"}:
        raise ValueError("split must be all, train, or eval")

    data_selection = load_shogi_policy_value_data_selection(data_selection_path)
    split_sources = (
        ("train", data_selection.train_sources),
        ("eval", data_selection.eval_sources),
    )
    tasks: list[ShogiTensorCacheTask] = []
    for split_name, sources in split_sources:
        if split != "all" and split != split_name:
            continue
        for source_index, source in enumerate(sources):
            example_count = count_shogi_policy_value_data_selection_source_examples(source)
            shard_index = 0
            for start in range(0, example_count, shard_examples):
                end = min(start + shard_examples, example_count)
                tasks.append(
                    {
                        "split": split_name,
                        "source_index": source_index,
                        "source_example_start_index": start,
                        "source_example_end_index": end,
                        "shard_index": shard_index,
                        "sample_count": end - start,
                    }
                )
                shard_index += 1
    return tasks


def count_shogi_policy_value_data_selection_source_examples(
    source: ShogiPolicyValueDataSelectionSource,
) -> int:
    if source.kind == "shogi_policy_value_examples_jsonl":
        example_count = _count_jsonl_records(source.path)
    elif source.kind == "game_records_jsonl":
        example_count = _count_game_record_examples(source.path, max_games=source.max_games)
    else:
        raise ValueError(f"unsupported data selection source kind: {source.kind}")

    if source.max_examples is not None:
        example_count = min(example_count, source.max_examples)
    return example_count


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _count_game_record_examples(path: Path, *, max_games: int | None) -> int:
    count = 0
    game_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            move_count = _game_record_move_count(json.loads(stripped))
            if move_count == 0:
                continue
            if max_games is not None and game_count >= max_games:
                break
            count += move_count
            game_count += 1
    return count


def _game_record_move_count(payload: object) -> int:
    if not isinstance(payload, dict):
        raise ValueError("shogi game record must be an object")
    if "moves" in payload:
        moves = payload["moves"]
        if not isinstance(moves, list):
            raise ValueError("shogi game record moves must be a list")
        return len(moves)
    if "transitions" in payload:
        transitions = payload["transitions"]
        if not isinstance(transitions, list):
            raise ValueError("shogi game record transitions must be a list")
        return len(transitions)
    raise ValueError("shogi game record must contain moves")
