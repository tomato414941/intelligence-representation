from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from intrep.domains.shogi.game_record import ShogiGameRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Split shogi game records at game boundaries.")
    parser.add_argument("--games-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=args.games_jsonl,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(json.dumps({"train_games": train_count, "eval_games": eval_count}))


def split_shogi_game_records_jsonl(
    *,
    games_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    eval_ratio: float,
    seed: int = 7,
) -> tuple[int, int]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval-ratio must be between 0 and 1")
    line_infos = _read_game_record_line_infos(games_jsonl)
    if len(line_infos) < 2:
        raise ValueError("at least two games are required to split")

    train_indices, eval_indices = _split_record_indices(line_infos, eval_ratio=eval_ratio, seed=seed)

    train_count, eval_count = _copy_split_jsonl_lines(
        games_jsonl=games_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        line_infos=line_infos,
        train_indices=train_indices,
        eval_indices=eval_indices,
    )
    return train_count, eval_count


def split_shogi_game_records(
    records: list[ShogiGameRecord],
    *,
    eval_ratio: float,
    seed: int = 7,
) -> tuple[list[ShogiGameRecord], list[ShogiGameRecord]]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval-ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("at least two games are required to split")

    train_records: list[ShogiGameRecord] = []
    eval_records: list[ShogiGameRecord] = []
    for actor_pair, group in sorted(_records_by_actor_pair(records).items()):
        shuffled = list(group)
        random.Random(f"{seed}:{actor_pair}").shuffle(shuffled)
        eval_count = _eval_count(len(shuffled), eval_ratio)
        train_records.extend(shuffled[:-eval_count])
        eval_records.extend(shuffled[-eval_count:])
    if not train_records or not eval_records:
        raise ValueError("split must produce non-empty train and eval games")
    return train_records, eval_records


def _records_by_actor_pair(records: list[ShogiGameRecord]) -> dict[str, list[ShogiGameRecord]]:
    groups: dict[str, list[ShogiGameRecord]] = {}
    for record in records:
        key = f"{record.black_actor.kind}:{record.white_actor.kind}"
        groups.setdefault(key, []).append(record)
    return groups


def _read_game_record_line_infos(path: Path) -> list[tuple[int, int, str]]:
    infos: list[tuple[int, int, str]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not _payload_has_moves(payload):
                continue
            infos.append((len(infos), line_number, _actor_pair_from_payload(payload)))
    return infos


def _split_record_indices(
    line_infos: list[tuple[int, int, str]],
    *,
    eval_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    groups: dict[str, list[int]] = {}
    for record_index, _line_number, actor_pair in line_infos:
        groups.setdefault(actor_pair, []).append(record_index)

    train_indices: set[int] = set()
    eval_indices: set[int] = set()
    for actor_pair, group in sorted(groups.items()):
        shuffled = list(group)
        random.Random(f"{seed}:{actor_pair}").shuffle(shuffled)
        eval_count = _eval_count(len(shuffled), eval_ratio)
        train_indices.update(shuffled[:-eval_count])
        eval_indices.update(shuffled[-eval_count:])
    if not train_indices or not eval_indices:
        raise ValueError("split must produce non-empty train and eval games")
    return train_indices, eval_indices


def _record_indices_to_line_numbers(line_infos: list[tuple[int, int, str]], record_indices: set[int]) -> set[int]:
    return {line_number for record_index, line_number, _actor_pair in line_infos if record_index in record_indices}


def _copy_split_jsonl_lines(
    *,
    games_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    line_infos: list[tuple[int, int, str]],
    train_indices: set[int],
    eval_indices: set[int],
) -> tuple[int, int]:
    train_jsonl.parent.mkdir(parents=True, exist_ok=True)
    eval_jsonl.parent.mkdir(parents=True, exist_ok=True)
    train_line_numbers = _record_indices_to_line_numbers(line_infos, train_indices)
    eval_line_numbers = _record_indices_to_line_numbers(line_infos, eval_indices)
    train_count = 0
    eval_count = 0
    with (
        games_jsonl.open(encoding="utf-8") as source,
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
    if train_count != len(train_indices) or eval_count != len(eval_indices):
        raise RuntimeError("split output counts did not match assigned record counts")
    return train_count, eval_count


def _actor_pair_from_payload(payload: dict[str, object]) -> str:
    black_actor = _object_dict(payload.get("black_actor"))
    white_actor = _object_dict(payload.get("white_actor"))
    return f"{black_actor['kind']}:{white_actor['kind']}"


def _payload_has_moves(payload: dict[str, object]) -> bool:
    moves = payload.get("moves")
    if isinstance(moves, list):
        return bool(moves)
    transitions = payload.get("transitions")
    if isinstance(transitions, list):
        return bool(transitions)
    return False


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _eval_count(game_count: int, eval_ratio: float) -> int:
    if game_count == 1:
        return 0
    count = max(1, round(game_count * eval_ratio))
    return min(count, game_count - 1)


if __name__ == "__main__":
    main()
