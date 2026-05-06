from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from intrep.tasks.shogi_move_choice.data import shogi_move_choice_examples_from_game_record
from intrep.tasks.shogi_move_choice.examples import ShogiMoveChoiceExample
from intrep.worlds.shogi.game_record import ShogiGameRecord, iter_shogi_game_records_jsonl
from intrep.worlds.shogi.game_replay import replay_shogi_game_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shogi move-choice examples from game records.")
    parser.add_argument("--games-jsonl", type=Path, required=True)
    parser.add_argument("--examples-jsonl", type=Path, required=True)
    parser.add_argument("--failures-jsonl", type=Path)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--include-actor-kind")
    parser.add_argument("--include-actor-name")
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be between 0 and shard-count - 1")

    game_count = 0
    failed_game_count = 0
    example_count = 0
    input_game_count = 0
    args.examples_jsonl.parent.mkdir(parents=True, exist_ok=True)
    failure_output = None
    if args.failures_jsonl is not None:
        args.failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
        failure_output = args.failures_jsonl.open("w", encoding="utf-8")
    with args.examples_jsonl.open("w", encoding="utf-8") as output:
        for input_index, record in enumerate(iter_shogi_game_records_jsonl(args.games_jsonl)):
            input_game_count += 1
            if input_index % args.shard_count != args.shard_index:
                continue
            if args.max_games is not None and game_count >= args.max_games:
                break
            try:
                examples = shogi_move_choice_examples_from_game_record(record)
                examples = _filter_examples_by_source_actor(
                    record,
                    examples,
                    include_actor_kind=args.include_actor_kind,
                    include_actor_name=args.include_actor_name,
                )
            except Exception as error:
                if failure_output is None:
                    raise
                failed_game_count += 1
                failure_output.write(
                    json.dumps(
                        {
                            "input_index": input_index,
                            "shard_index": args.shard_index,
                            "shard_count": args.shard_count,
                            "winner": record.winner,
                            "actions": [transition.action_usi for transition in record.transitions],
                            "error": type(error).__name__,
                            "message": str(error),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                continue
            for ply_index, example in enumerate(examples):
                output.write(
                    json.dumps(
                        {
                            "position_sfen": example.position_sfen,
                            "legal_moves": list(example.legal_moves),
                            "chosen_move": example.chosen_move,
                            "value_target": example.value_target,
                            "game_index": input_index,
                            "ply_index": example.ply_index if example.ply_index is not None else ply_index,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            game_count += 1
            example_count += len(examples)
            if args.progress_every > 0 and game_count % args.progress_every == 0:
                print(f"games={game_count} examples={example_count}", flush=True)
    if failure_output is not None:
        failure_output.close()

    if example_count == 0:
        raise ValueError("no examples were written")
    print(
        json.dumps(
            {
                "input_games_seen": input_game_count,
                "games": game_count,
                "failed_games": failed_game_count,
                "examples": example_count,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "examples_jsonl": str(args.examples_jsonl),
            }
        )
    )


def _filter_examples_by_source_actor(
    record: ShogiGameRecord,
    examples: list[ShogiMoveChoiceExample],
    *,
    include_actor_kind: str | None,
    include_actor_name: str | None,
) -> list[ShogiMoveChoiceExample]:
    if include_actor_kind is None and include_actor_name is None:
        return examples
    plies = replay_shogi_game_record(record)
    if len(plies) != len(examples):
        raise ValueError("replayed ply count must match example count")
    filtered: list[ShogiMoveChoiceExample] = []
    for ply, example in zip(plies, examples):
        if include_actor_kind is not None and ply.source_actor.kind != include_actor_kind:
            continue
        if include_actor_name is not None and ply.source_actor.name != include_actor_name:
            continue
        filtered.append(replace(example, ply_index=ply.ply_index))
    return filtered


if __name__ == "__main__":
    main()
