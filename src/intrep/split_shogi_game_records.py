from __future__ import annotations

import argparse
import json
from pathlib import Path

from intrep.shogi_game_record import iter_shogi_game_records_jsonl, write_shogi_game_records_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split shogi game records at game boundaries.")
    parser.add_argument("--games-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=args.games_jsonl,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        eval_ratio=args.eval_ratio,
    )
    print(json.dumps({"train_games": train_count, "eval_games": eval_count}))


def split_shogi_game_records_jsonl(
    *,
    games_jsonl: Path,
    train_jsonl: Path,
    eval_jsonl: Path,
    eval_ratio: float,
) -> tuple[int, int]:
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError("eval-ratio must be between 0 and 1")
    records = list(iter_shogi_game_records_jsonl(games_jsonl))
    if len(records) < 2:
        raise ValueError("at least two games are required to split")

    eval_count = max(1, round(len(records) * eval_ratio))
    if eval_count >= len(records):
        raise ValueError("eval split must leave at least one train game")
    train_count = len(records) - eval_count

    write_shogi_game_records_jsonl(train_jsonl, records[:train_count])
    write_shogi_game_records_jsonl(eval_jsonl, records[train_count:])
    return train_count, eval_count


if __name__ == "__main__":
    main()
