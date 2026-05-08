from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from intrep.worlds.shogi.experience_stats import shogi_position_stats, shogi_train_eval_position_stats
from intrep.worlds.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl, write_shogi_game_records_jsonl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a fixed shogi training view from explicit train/eval game logs.")
    parser.add_argument("--train-games", type=Path, required=True)
    parser.add_argument("--eval-games", type=Path, required=True)
    parser.add_argument("--max-train-games", type=int)
    parser.add_argument("--max-eval-games", type=int)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/shogi/datasets"))
    parser.add_argument("--policy-target-source", choices=("chosen_move", "usi_multipv"), default="chosen_move")
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--policy-mate-cp", type=float, default=100000.0)
    parser.add_argument("--value-target-source", choices=("winner", "yaneuraou_best_score"), default="winner")
    parser.add_argument("--score-cp-scale", type=float, default=600.0)
    args = parser.parse_args(argv)

    result = create_shogi_training_view(
        train_games=args.train_games,
        eval_games=args.eval_games,
        max_train_games=args.max_train_games,
        max_eval_games=args.max_eval_games,
        name=args.name,
        output_root=args.output_root,
        policy_target_source=args.policy_target_source,
        policy_temperature_cp=args.policy_temperature_cp,
        policy_mate_cp=args.policy_mate_cp,
        value_target_source=args.value_target_source,
        score_cp_scale=args.score_cp_scale,
    )
    print(json.dumps(result, indent=2))


def create_shogi_training_view(
    *,
    train_games: Path,
    eval_games: Path,
    name: str,
    output_root: Path,
    max_train_games: int | None = None,
    max_eval_games: int | None = None,
    policy_target_source: str = "chosen_move",
    policy_temperature_cp: float = 100.0,
    policy_mate_cp: float = 100000.0,
    value_target_source: str = "winner",
    score_cp_scale: float = 600.0,
) -> dict[str, object]:
    output_dir = output_root / name
    games_jsonl = output_dir / "games.jsonl"
    train_jsonl = output_dir / "train-games.jsonl"
    eval_jsonl = output_dir / "eval-games.jsonl"
    data_selection_json = output_dir / "data-selection.json"
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists():
        raise FileExistsError(f"training view already exists: {output_dir}")
    _validate_max_games(max_train_games, label="max_train_games")
    _validate_max_games(max_eval_games, label="max_eval_games")

    train_records = _limit_records(list(iter_shogi_game_records_jsonl(train_games)), max_train_games)
    eval_records = _limit_records(list(iter_shogi_game_records_jsonl(eval_games)), max_eval_games)
    if not train_records:
        raise ValueError("train games must not be empty")
    if not eval_records:
        raise ValueError("eval games must not be empty")
    records = train_records + eval_records
    train_eval_position_stats = shogi_train_eval_position_stats(train_records, eval_records).to_dict()

    output_dir.mkdir(parents=True)
    write_shogi_game_records_jsonl(games_jsonl, records)
    write_shogi_game_records_jsonl(train_jsonl, train_records)
    write_shogi_game_records_jsonl(eval_jsonl, eval_records)

    data_selection = {
        "name": name,
        "objective": "shogi move-choice policy/value",
        "policy_target_source": policy_target_source,
        "policy_temperature_cp": policy_temperature_cp,
        "policy_mate_cp": policy_mate_cp,
        "value_target_source": value_target_source,
        "score_cp_scale": score_cp_scale,
        "train_sources": [_source_json(train_jsonl.name, max_train_games)],
        "eval_sources": [_source_json(eval_jsonl.name, max_eval_games)],
    }
    data_selection_json.write_text(json.dumps(data_selection, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "shogi_training_view_v1",
        "record_schema": "shogi_game_record_jsonl",
        "name": name,
        "created_at": datetime.now(UTC).isoformat(),
        "train_source_games_jsonl": str(train_games),
        "eval_source_games_jsonl": str(eval_games),
        "max_train_games": max_train_games,
        "max_eval_games": max_eval_games,
        "game_count": len(records),
        "transition_count": sum(len(record.transitions) for record in records),
        "position_stats": shogi_position_stats(records).to_dict(),
        **train_eval_position_stats,
        "actor_pair_counts": _actor_pair_counts(records),
        "train_actor_pair_counts": _actor_pair_counts(train_records),
        "eval_actor_pair_counts": _actor_pair_counts(eval_records),
        "train_games": len(train_records),
        "eval_games": len(eval_records),
        "policy_target_source": policy_target_source,
        "policy_temperature_cp": policy_temperature_cp,
        "policy_mate_cp": policy_mate_cp,
        "value_target_source": value_target_source,
        "score_cp_scale": score_cp_scale,
        "files": {
            "games": games_jsonl.name,
            "train": train_jsonl.name,
            "eval": eval_jsonl.name,
            "data_selection": data_selection_json.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "training_view": str(output_dir),
        "data_selection_json": str(data_selection_json),
        "games_jsonl": str(games_jsonl),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
        "manifest": str(manifest_path),
        "game_count": len(records),
        "train_games": len(train_records),
        "eval_games": len(eval_records),
    }


def _actor_pair_counts(records: list[ShogiGameRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record.black_actor.kind}:{record.white_actor.kind}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _limit_records(records: list[ShogiGameRecord], max_games: int | None) -> list[ShogiGameRecord]:
    if max_games is None:
        return records
    return records[:max_games]


def _source_json(path: str, max_games: int | None) -> dict[str, str | int]:
    payload: dict[str, str | int] = {"kind": "game_records_jsonl", "path": path}
    if max_games is not None:
        payload["max_games"] = max_games
    return payload


def _validate_max_games(value: int | None, *, label: str) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{label} must be positive")


if __name__ == "__main__":
    main()
