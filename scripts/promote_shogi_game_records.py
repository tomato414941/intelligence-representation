from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from intrep.worlds.shogi.game_record import iter_shogi_game_records_jsonl, write_shogi_game_records_jsonl
from intrep.worlds.shogi.game_split import split_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Promote shogi game records into a managed record collection.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/records"))
    parser.add_argument("--eval-ratio", type=float, default=0.25)
    args = parser.parse_args(argv)

    result = promote_shogi_game_records(
        input_path=args.input,
        name=args.name,
        output_root=args.output_root,
        eval_ratio=args.eval_ratio,
    )
    print(json.dumps(result, indent=2))


def promote_shogi_game_records(
    *,
    input_path: Path,
    name: str,
    output_root: Path,
    eval_ratio: float,
) -> dict[str, object]:
    output_dir = output_root / name
    games_jsonl = output_dir / "games.jsonl"
    train_jsonl = output_dir / "train-games.jsonl"
    eval_jsonl = output_dir / "eval-games.jsonl"
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists():
        raise FileExistsError(f"record collection already exists: {output_dir}")

    records = list(iter_shogi_game_records_jsonl(input_path))
    if not records:
        raise ValueError("input must contain at least one non-empty shogi game record")
    write_shogi_game_records_jsonl(games_jsonl, records)
    train_count, eval_count = split_shogi_game_records_jsonl(
        games_jsonl=games_jsonl,
        train_jsonl=train_jsonl,
        eval_jsonl=eval_jsonl,
        eval_ratio=eval_ratio,
    )
    transition_count = sum(len(record.transitions) for record in records)
    manifest = {
        "schema": "shogi_game_record_collection_v1",
        "record_schema": "shogi_game_record_jsonl",
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(input_path),
        "game_count": len(records),
        "transition_count": transition_count,
        "train_games": train_count,
        "eval_games": eval_count,
        "eval_ratio": eval_ratio,
        "files": {
            "games": games_jsonl.name,
            "train": train_jsonl.name,
            "eval": eval_jsonl.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "record_collection": str(output_dir),
        "games_jsonl": str(games_jsonl),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "manifest": str(manifest_path),
        "game_count": len(records),
        "transition_count": transition_count,
        "train_games": train_count,
        "eval_games": eval_count,
    }


if __name__ == "__main__":
    main()
