from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from intrep.domains.shogi.game_record import (
    iter_shogi_game_records_jsonl,
    shogi_game_record_to_json,
)


def archive_shogi_generated_records(
    *,
    input_path: Path,
    output_root: Path,
    record_set_id: str,
    source_run: str | None = None,
    generation_method: str | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    output_dir = output_root / record_set_id
    games_path = output_dir / "games.jsonl"
    manifest_path = output_dir / "manifest.json"
    _ensure_writable((games_path, manifest_path), overwrite=overwrite)

    output_dir.mkdir(parents=True, exist_ok=True)
    game_count = 0
    transition_count = 0
    with games_path.open("w", encoding="utf-8") as games_out:
        for record in iter_shogi_game_records_jsonl(input_path):
            games_out.write(json.dumps(shogi_game_record_to_json(record), separators=(",", ":"), sort_keys=True))
            games_out.write("\n")
            game_count += 1
            transition_count += len(record.moves)
    if game_count == 0:
        raise ValueError("input must contain at least one non-empty shogi game record")

    manifest: dict[str, object] = {
        "schema": "intrep.shogi_generated_record_archive.v1",
        "record_schema": "shogi_game_record_jsonl",
        "name": record_set_id,
        "created_at": datetime.now(UTC).isoformat(),
        "source_name": "generated",
        "source_path": str(input_path),
        "source_run": source_run,
        "generation_method": generation_method,
        "game_count": game_count,
        "transition_count": transition_count,
        "files": {
            "games": games_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "games": str(games_path),
        "manifest": str(manifest_path),
        "game_count": game_count,
        "transition_count": transition_count,
    }


def _ensure_writable(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"output already exists: {', '.join(str(path) for path in existing)}")
