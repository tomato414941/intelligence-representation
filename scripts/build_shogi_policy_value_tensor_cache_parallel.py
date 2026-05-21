from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from intrep.problems.shogi_policy_value.output_space import (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACES,
)
from intrep.problems.shogi_policy_value.tensor_cache import (
    build_shogi_policy_value_tensor_cache_shard,
    write_shogi_policy_value_tensor_cache_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shogi policy/value tensor cache with shard parallelism.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--output-space",
        choices=SHOGI_POLICY_VALUE_OUTPUT_SPACES,
        default=SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    )
    parser.add_argument("--shard-examples", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    tasks = _build_tasks(
        data_selection_path=args.data_selection,
        shard_examples=args.shard_examples,
    )
    print(
        json.dumps(
            {
                "event": "build_start",
                "data_selection": str(args.data_selection),
                "cache_dir": str(args.out),
                "output_space": args.output_space,
                "shard_examples": args.shard_examples,
                "jobs": args.jobs,
                "shard_count": len(tasks),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    completed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                _build_shard_task,
                task,
                data_selection_path=args.data_selection,
                cache_dir=args.out,
                output_space=args.output_space,
                resume=args.resume,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "shard_done",
                        "completed_shards": completed,
                        "shard_count": len(tasks),
                        "split": result["split"],
                        "source_index": result["source_index"],
                        "source_example_start_index": result["source_example_start_index"],
                        "source_example_end_index": result["source_example_end_index"],
                        "sample_count": result["sample_count"],
                        "path": result["path"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    manifest = write_shogi_policy_value_tensor_cache_manifest(
        data_selection_path=args.data_selection,
        cache_dir=args.out,
        shard_examples=args.shard_examples,
        output_space=args.output_space,
    )
    print(
        json.dumps(
            {
                "event": "manifest_done",
                "train_count": manifest["train_count"],
                "eval_count": manifest["eval_count"],
                "shard_count": len(manifest["shards"]),
                "cache_dir": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _build_shard_task(
    task: dict[str, int | str],
    *,
    data_selection_path: Path,
    cache_dir: Path,
    output_space: str,
    resume: bool,
) -> dict[str, object]:
    return build_shogi_policy_value_tensor_cache_shard(
        data_selection_path=data_selection_path,
        cache_dir=cache_dir,
        split=str(task["split"]),
        source_index=int(task["source_index"]),
        source_example_start_index=int(task["source_example_start_index"]),
        source_example_end_index=int(task["source_example_end_index"]),
        shard_index=int(task["shard_index"]),
        resume=resume,
        output_space=output_space,
    )


def _build_tasks(*, data_selection_path: Path, shard_examples: int) -> list[dict[str, int | str]]:
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    payload = json.loads(data_selection_path.read_text(encoding="utf-8"))
    root = data_selection_path.parent
    tasks: list[dict[str, int | str]] = []
    for split in ("train", "eval"):
        for source_index, source in enumerate(_object_list(payload[f"{split}_sources"])):
            source_payload = _object_dict(source)
            source_path = Path(str(source_payload["path"]))
            if not source_path.is_absolute():
                source_path = root / source_path
            example_count = _count_jsonl_records(source_path)
            if "max_examples" in source_payload:
                example_count = min(example_count, int(source_payload["max_examples"]))
            shard_index = 0
            for start in range(0, example_count, shard_examples):
                end = min(start + shard_examples, example_count)
                tasks.append(
                    {
                        "split": split,
                        "source_index": source_index,
                        "source_example_start_index": start,
                        "source_example_end_index": end,
                        "shard_index": shard_index,
                    }
                )
                shard_index += 1
    return tasks


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _object_dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


if __name__ == "__main__":
    main()
