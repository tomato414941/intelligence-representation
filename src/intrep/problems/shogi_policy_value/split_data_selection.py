from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Iterable

from intrep.problems.shogi_policy_value.data_selection import (
    ShogiPolicyValueDataSelection,
    ShogiPolicyValueDataSelectionSource,
    load_shogi_policy_value_data_selection,
    shogi_policy_value_data_selection_to_json,
)
from intrep.experience.shogi.game_split import split_shogi_game_records_jsonl


@dataclass(frozen=True)
class ShogiPolicyValueSplitSourceResult:
    source_role: str
    source_index: int
    kind: str
    source_path: str
    train_path: str
    eval_path: str
    train_count: int
    eval_count: int


@dataclass(frozen=True)
class ShogiPolicyValueSplitDataSelectionResult:
    data_selection_path: Path
    manifest_path: Path
    train_count: int
    eval_count: int
    source_results: tuple[ShogiPolicyValueSplitSourceResult, ...]


def split_shogi_policy_value_data_selection(
    *,
    data_selection_path: Path,
    output_dir: Path,
    name: str | None = None,
    eval_ratio: float = 0.05,
    seed: int = 7,
) -> ShogiPolicyValueSplitDataSelectionResult:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    selection = load_shogi_policy_value_data_selection(data_selection_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    source_results: list[ShogiPolicyValueSplitSourceResult] = []
    train_sources: list[ShogiPolicyValueDataSelectionSource] = []
    eval_sources: list[ShogiPolicyValueDataSelectionSource] = []
    pooled_sources = [
        *(
            ("train", index, source)
            for index, source in enumerate(selection.train_sources)
        ),
        *(
            ("eval", index, source)
            for index, source in enumerate(selection.eval_sources)
        ),
    ]
    for pooled_index, (source_role, source_index, source) in enumerate(pooled_sources):
        train_path, eval_path, train_count, eval_count = _split_source(
            source,
            output_dir=splits_dir,
            source_index=pooled_index,
            eval_ratio=eval_ratio,
            seed=seed,
        )
        train_sources.append(
            ShogiPolicyValueDataSelectionSource(
                kind=source.kind,
                path=train_path,
            )
        )
        eval_sources.append(
            ShogiPolicyValueDataSelectionSource(
                kind=source.kind,
                path=eval_path,
            )
        )
        source_results.append(
            ShogiPolicyValueSplitSourceResult(
                source_role=source_role,
                source_index=source_index,
                kind=source.kind,
                source_path=str(source.path),
                train_path=str(train_path),
                eval_path=str(eval_path),
                train_count=train_count,
                eval_count=eval_count,
            )
        )

    output_selection = ShogiPolicyValueDataSelection(
        name=name or selection.name,
        objective=selection.objective,
        target_construction=selection.target_construction,
        analysis_sources=selection.analysis_sources,
        train_sources=tuple(train_sources),
        eval_sources=tuple(eval_sources),
    )
    output_data_selection_path = output_dir / "data-selection.json"
    output_data_selection_path.write_text(
        json.dumps(shogi_policy_value_data_selection_to_json(output_selection, root=output_dir), indent=2) + "\n",
        encoding="utf-8",
    )
    train_count_total = sum(result.train_count for result in source_results)
    eval_count_total = sum(result.eval_count for result in source_results)
    manifest_path = output_dir / "split-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "intrep.shogi_policy_value_split_data_selection.v1",
                "source_data_selection_path": str(data_selection_path),
                "source_data_selection_name": selection.name,
                "data_selection_path": str(output_data_selection_path),
                "name": output_selection.name,
                "eval_ratio": eval_ratio,
                "seed": seed,
                "train_count": train_count_total,
                "eval_count": eval_count_total,
                "sources": [asdict(result) for result in source_results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ShogiPolicyValueSplitDataSelectionResult(
        data_selection_path=output_data_selection_path,
        manifest_path=manifest_path,
        train_count=train_count_total,
        eval_count=eval_count_total,
        source_results=tuple(source_results),
    )


def _split_source(
    source: ShogiPolicyValueDataSelectionSource,
    *,
    output_dir: Path,
    source_index: int,
    eval_ratio: float,
    seed: int,
) -> tuple[Path, Path, int, int]:
    if source.max_games is not None or source.max_examples is not None:
        raise ValueError("limited data selection sources are not supported by split_data_selection")
    if source.kind == "game_records_jsonl":
        train_path = output_dir / f"source-{source_index:04d}-train-games.jsonl"
        eval_path = output_dir / f"source-{source_index:04d}-eval-games.jsonl"
        split_shogi_game_records_jsonl(
            games_jsonl=source.path,
            train_jsonl=train_path,
            eval_jsonl=eval_path,
            eval_ratio=eval_ratio,
            seed=seed,
        )
        train_count = _count_game_record_examples(train_path)
        eval_count = _count_game_record_examples(eval_path)
        return train_path, eval_path, train_count, eval_count
    if source.kind == "shogi_policy_value_examples_jsonl":
        train_path = output_dir / f"source-{source_index:04d}-train-examples.jsonl"
        eval_path = output_dir / f"source-{source_index:04d}-eval-examples.jsonl"
        train_count, eval_count = _split_examples_jsonl(
            examples_jsonl=source.path,
            train_jsonl=train_path,
            eval_jsonl=eval_path,
            eval_ratio=eval_ratio,
            seed=seed,
        )
        return train_path, eval_path, train_count, eval_count
    raise ValueError(f"unsupported data selection source kind: {source.kind}")


def _split_examples_jsonl(
    *,
    examples_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    eval_ratio: float,
    seed: int,
) -> tuple[int, int]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval_ratio must be between 0 and 1")
    line_infos = _read_example_line_infos(examples_jsonl)
    if len({game_index for _line_number, game_index in line_infos}) < 2:
        raise ValueError("at least two games are required to split examples")
    train_game_indices, eval_game_indices = _split_game_indices(
        (game_index for _line_number, game_index in line_infos),
        eval_ratio=eval_ratio,
        seed=seed,
    )
    train_line_numbers = {
        line_number for line_number, game_index in line_infos if game_index in train_game_indices
    }
    eval_line_numbers = {
        line_number for line_number, game_index in line_infos if game_index in eval_game_indices
    }
    return _copy_example_split_lines(
        examples_jsonl=examples_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        train_line_numbers=train_line_numbers,
        eval_line_numbers=eval_line_numbers,
    )


def _read_example_line_infos(path: Path) -> list[tuple[int, int]]:
    infos: list[tuple[int, int]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("shogi policy/value example must be an object")
            game_index = payload.get("game_index")
            if game_index is None:
                raise ValueError(f"example is missing game_index: {path}:{line_number + 1}")
            infos.append((line_number, int(game_index)))
    return infos


def _count_game_record_examples(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError("shogi game record must be an object")
            transitions = payload.get("transitions")
            if isinstance(transitions, list):
                count += len(transitions)
                continue
            moves = payload.get("moves")
            if isinstance(moves, list):
                count += len(moves)
    return count


def _split_game_indices(
    game_indices: Iterable[int],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    unique_game_indices = sorted(set(game_indices))
    shuffled = list(unique_game_indices)
    random.Random(str(seed)).shuffle(shuffled)
    eval_count = _eval_count(len(shuffled), eval_ratio)
    train_indices = set(shuffled[:-eval_count])
    eval_indices = set(shuffled[-eval_count:])
    if not train_indices or not eval_indices:
        raise ValueError("split must produce non-empty train and eval examples")
    return train_indices, eval_indices


def _copy_example_split_lines(
    *,
    examples_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    train_line_numbers: set[int],
    eval_line_numbers: set[int],
) -> tuple[int, int]:
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_count = 0
    eval_count = 0
    with (
        examples_jsonl.open(encoding="utf-8") as source,
        train_jsonl.open("w", encoding="utf-8") as train_out,
        eval_jsonl.open("w", encoding="utf-8") as eval_out,
    ):
        for line_number, line in enumerate(source):
            if line_number in train_line_numbers:
                train_out.write(line if line.endswith("\n") else line + "\n")
                train_count += 1
            elif line_number in eval_line_numbers:
                eval_out.write(line if line.endswith("\n") else line + "\n")
                eval_count += 1
    if train_count != len(train_line_numbers) or eval_count != len(eval_line_numbers):
        raise RuntimeError("split output counts did not match assigned example counts")
    return train_count, eval_count


def _eval_count(game_count: int, eval_ratio: float) -> int:
    if game_count < 2:
        return 0
    count = max(1, round(game_count * eval_ratio))
    return min(count, game_count - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split all shogi policy/value data selection sources into train/eval files.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = split_shogi_policy_value_data_selection(
        data_selection_path=args.data_selection,
        output_dir=args.output_dir,
        name=args.name,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "data_selection": str(result.data_selection_path),
                "manifest": str(result.manifest_path),
                "train_count": result.train_count,
                "eval_count": result.eval_count,
            }
        )
    )


if __name__ == "__main__":
    main()
