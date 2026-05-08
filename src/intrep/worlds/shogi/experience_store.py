from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from intrep.worlds.shogi.experience_stats import shogi_position_stats
from intrep.worlds.shogi.game_record import (
    ShogiGameRecord,
    iter_shogi_game_records_jsonl,
    write_shogi_game_records_jsonl,
)
from intrep.worlds.shogi.source_selection import shogi_actor_pair_counts


def append_shogi_experience_store(*, input_path: Path, store_dir: Path) -> dict[str, object]:
    games_jsonl = store_dir / "games.jsonl"
    manifest_path = store_dir / "manifest.json"
    history_path = store_dir / "history.jsonl"

    existing_records = list(iter_shogi_game_records_jsonl(games_jsonl)) if games_jsonl.exists() else []
    new_records = list(iter_shogi_game_records_jsonl(input_path))
    if not new_records:
        raise ValueError("input must contain at least one non-empty shogi game record")

    all_records = existing_records + new_records
    write_shogi_game_records_jsonl(games_jsonl, all_records)
    total_position_stats = shogi_position_stats(all_records).to_dict()

    event = {
        "event": "append",
        "created_at": datetime.now(UTC).isoformat(),
        "source_path": str(input_path),
        "added_games": len(new_records),
        "added_transitions": _transition_count(new_records),
        "added_actor_pair_counts": shogi_actor_pair_counts(new_records),
        "added_checkpoint_actor_counts": _checkpoint_actor_counts(new_records),
        "total_games": len(all_records),
        "total_transitions": _transition_count(all_records),
        "total_actor_pair_counts": shogi_actor_pair_counts(all_records),
        "total_checkpoint_actor_counts": _checkpoint_actor_counts(all_records),
        "total_position_stats": total_position_stats,
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")

    manifest = {
        "schema": "shogi_experience_store_v1",
        "record_schema": "shogi_game_record_jsonl",
        "updated_at": event["created_at"],
        "game_count": len(all_records),
        "transition_count": event["total_transitions"],
        "position_stats": total_position_stats,
        "actor_pair_counts": event["total_actor_pair_counts"],
        "checkpoint_actor_counts": event["total_checkpoint_actor_counts"],
        "files": {
            "games": games_jsonl.name,
            "history": history_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "store": str(store_dir),
        "games_jsonl": str(games_jsonl),
        "manifest": str(manifest_path),
        "history": str(history_path),
        "added_games": len(new_records),
        "total_games": len(all_records),
        "total_transitions": event["total_transitions"],
    }


def _transition_count(records: list[ShogiGameRecord]) -> int:
    return sum(len(record.transitions) for record in records)


def _checkpoint_actor_counts(records: list[ShogiGameRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        for actor in (record.black_actor, record.white_actor):
            if actor.kind != "checkpoint":
                continue
            key = _checkpoint_actor_key(actor.settings)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _checkpoint_actor_key(settings: dict[str, str | int | float | bool | None]) -> str:
    checkpoint = settings.get("checkpoint", "unknown")
    policy = settings.get("policy", "unknown")
    simulations = settings.get("simulations", "unknown")
    return f"{checkpoint} | policy={policy} | simulations={simulations}"
