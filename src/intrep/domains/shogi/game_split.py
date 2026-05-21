from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from intrep.domains.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl, write_shogi_game_records_jsonl


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
    records = list(iter_shogi_game_records_jsonl(games_jsonl))
    if len(records) < 2:
        raise ValueError("at least two games are required to split")

    train_records, eval_records = split_shogi_game_records(records, eval_ratio=eval_ratio, seed=seed)

    write_shogi_game_records_jsonl(train_jsonl, train_records)
    write_shogi_game_records_jsonl(eval_jsonl, eval_records)
    return len(train_records), len(eval_records)


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


def _eval_count(game_count: int, eval_ratio: float) -> int:
    if game_count == 1:
        return 0
    count = max(1, round(game_count * eval_ratio))
    return min(count, game_count - 1)


if __name__ == "__main__":
    main()
