from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from intrep.problems.shogi_policy_value.output_space import shogi_policy_value_output_space_for_assembly_spec
from intrep.problems.shogi_policy_value.tensor_cache import (
    build_shogi_policy_value_tensor_cache_shard,
    write_shogi_policy_value_tensor_cache_manifest,
)
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS,
    shogi_policy_value_input_for_assembly_spec_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a shogi policy/value tensor cache with shard parallelism.")
    parser.add_argument("--data-selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--assembly-spec",
        choices=SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS,
        required=True,
    )
    parser.add_argument("--shard-examples", type=int, default=10_000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    input_module = shogi_policy_value_input_for_assembly_spec_id(args.assembly_spec)
    output_space = shogi_policy_value_output_space_for_assembly_spec(args.assembly_spec)
    tasks = _build_tasks(
        data_selection_path=args.data_selection,
        shard_examples=args.shard_examples,
    )
    total_expected_samples = sum(int(task["sample_count"]) for task in tasks)
    started = time.monotonic()
    print(
        json.dumps(
            {
                "event": "build_start",
                "data_selection": str(args.data_selection),
                "cache_dir": str(args.out),
                "assembly_spec": args.assembly_spec,
                "input_module": input_module,
                "output_space": output_space,
                "shard_examples": args.shard_examples,
                "jobs": args.jobs,
                "shard_count": len(tasks),
                "expected_sample_count": total_expected_samples,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    completed = 0
    completed_samples = 0
    completed_cache_bytes = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(
                _build_shard_task,
                task,
                data_selection_path=args.data_selection,
                cache_dir=args.out,
                output_space=output_space,
                input_module=input_module,
                resume=args.resume,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            completed_samples += int(result["sample_count"])
            cache_bytes = _shard_cache_bytes(args.out, str(result["path"]))
            completed_cache_bytes += cache_bytes
            elapsed_seconds = time.monotonic() - started
            samples_per_second = _rate(completed_samples, elapsed_seconds)
            remaining_samples = max(0, total_expected_samples - completed_samples)
            estimated_remaining_seconds = (
                remaining_samples / samples_per_second
                if samples_per_second > 0.0
                else None
            )
            print(
                json.dumps(
                    {
                        "event": "shard_done",
                        "completed_shards": completed,
                        "shard_count": len(tasks),
                        "completed_samples": completed_samples,
                        "expected_sample_count": total_expected_samples,
                        "split": result["split"],
                        "source_index": result["source_index"],
                        "source_example_start_index": result["source_example_start_index"],
                        "source_example_end_index": result["source_example_end_index"],
                        "sample_count": result["sample_count"],
                        "cache_bytes": cache_bytes,
                        "completed_cache_bytes": completed_cache_bytes,
                        "elapsed_seconds": elapsed_seconds,
                        "samples_per_second": samples_per_second,
                        "bytes_per_sample": _rate(completed_cache_bytes, completed_samples),
                        "estimated_remaining_seconds": estimated_remaining_seconds,
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
        output_space=output_space,
        input_module=input_module,
    )
    elapsed_seconds = time.monotonic() - started
    sample_count = int(manifest["train_count"]) + int(manifest["eval_count"])
    cache_bytes = _cache_bytes(args.out)
    summary = {
        "event": "build_summary",
        "data_selection": str(args.data_selection),
        "cache_dir": str(args.out),
        "assembly_spec": args.assembly_spec,
        "input_module": input_module,
        "output_space": output_space,
        "shard_examples": args.shard_examples,
        "jobs": args.jobs,
        "train_count": manifest["train_count"],
        "eval_count": manifest["eval_count"],
        "sample_count": sample_count,
        "shard_count": len(manifest["shards"]),
        "cache_bytes": cache_bytes,
        "pt_cache_bytes": _pt_cache_bytes(args.out),
        "bytes_per_sample": _rate(cache_bytes, sample_count),
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": _rate(sample_count, elapsed_seconds),
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            summary,
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
    input_module: str,
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
        input_module=input_module,
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
                        "sample_count": end - start,
                    }
                )
                shard_index += 1
    return tasks


def _shard_cache_bytes(cache_dir: Path, relative_path: str) -> int:
    path = cache_dir / relative_path
    return path.stat().st_size if path.exists() else 0


def _cache_bytes(cache_dir: Path) -> int:
    return sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())


def _pt_cache_bytes(cache_dir: Path) -> int:
    return sum(path.stat().st_size for path in cache_dir.glob("*/*.pt"))


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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
