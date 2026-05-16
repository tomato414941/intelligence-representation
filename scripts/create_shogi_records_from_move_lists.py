from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from intrep.worlds.shogi.game_record import shogi_game_record_to_json
from intrep.worlds.shogi.move_list_record import (
    iter_shogi_move_list_records_jsonl,
    shogi_game_record_from_move_list_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ShogiGameRecord JSONL from compact shogi move-list records.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--games-file", default="games.jsonl")
    parser.add_argument("--failures-file", default="failures.jsonl")
    parser.add_argument("--manifest-file", default="manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = create_shogi_records_from_move_lists(
        input_path=args.input,
        output_dir=args.output_dir,
        source_name=args.source_name,
        name=args.name,
        start_index=args.start_index,
        end_index=args.end_index,
        games_file=args.games_file,
        failures_file=args.failures_file,
        manifest_file=args.manifest_file,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))


def create_shogi_records_from_move_lists(
    *,
    input_path: Path,
    output_dir: Path,
    source_name: str,
    name: str,
    start_index: int = 0,
    end_index: int | None = None,
    games_file: str = "games.jsonl",
    failures_file: str = "failures.jsonl",
    manifest_file: str = "manifest.json",
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / games_file
    failures_path = output_dir / failures_file
    manifest_path = output_dir / manifest_file
    _ensure_writable((games_path, failures_path, manifest_path), overwrite=overwrite)

    game_count = 0
    transition_count = 0
    failure_count = 0
    first_source_record_index: int | None = None
    last_source_record_index: int | None = None

    with games_path.open("w", encoding="utf-8") as games_out, failures_path.open("w", encoding="utf-8") as failures_out:
        for source_record_index, move_list_record in iter_shogi_move_list_records_jsonl(
            input_path,
            start_index=start_index,
            end_index=end_index,
        ):
            try:
                game_record = shogi_game_record_from_move_list_record(
                    move_list_record,
                    source_name=source_name,
                    source_record_index=source_record_index,
                )
            except Exception as exc:  # noqa: BLE001
                failure_count += 1
                _write_jsonl(
                    failures_out,
                    {
                        "source_record_index": source_record_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                continue
            _write_jsonl(games_out, shogi_game_record_to_json(game_record))
            game_count += 1
            transition_count += len(game_record.moves)
            if first_source_record_index is None:
                first_source_record_index = source_record_index
            last_source_record_index = source_record_index

    manifest = {
        "schema": "shogi_game_record_collection_v1",
        "record_schema": "shogi_game_record_jsonl",
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "source_name": source_name,
        "source_path": str(input_path),
        "source_record_start_index": start_index,
        "source_record_end_index": end_index,
        "first_source_record_index": first_source_record_index,
        "last_source_record_index": last_source_record_index,
        "game_count": game_count,
        "transition_count": transition_count,
        "failure_count": failure_count,
        "files": {
            "games": games_file,
            "failures": failures_file,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "games": str(games_path),
        "failures": str(failures_path),
        "manifest": str(manifest_path),
        "game_count": game_count,
        "transition_count": transition_count,
        "failure_count": failure_count,
    }


def _write_jsonl(file, payload: dict[str, object]) -> None:
    file.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _ensure_writable(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {', '.join(str(path) for path in existing)}")


if __name__ == "__main__":
    main()
