from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - Modal is supplied by `modal run`.
    modal = None  # type: ignore[assignment]


APP_NAME = "intrep-shogi-policy-value-tensor-cache"
VOLUME_NAME = os.environ.get("INTREP_MODAL_VOLUME_NAME", "intrep-shogi-tensor-cache")
VOLUME_ROOT = Path("/data")
DEFAULT_LOCAL_BUNDLE = Path("data/shogi/training-data-bundles/qhapaq-full")
DEFAULT_REMOTE_BUNDLE = "qhapaq-full"
DEFAULT_CACHE_NAME = "shogi-policy-value-tensors"
DEFAULT_POLICY_PLANE_CACHE_NAME = "shogi-policy-plane-value-tensors"
DEFAULT_OUTPUT_SPACE = "candidate_move"


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy>=1.26", "python-shogi>=1.1.1", "tokenizers>=0.23.1", "torch>=2.2")
        .env({"PYTHONPATH": "/root/src"})
        .add_local_dir("src", "/root/src")
    )
    app = modal.App(APP_NAME)

    @app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=24 * 60 * 60)
    def build_remote_shard(task: dict[str, Any]) -> dict[str, object]:
        from intrep.problems.shogi_policy_value.tensor_cache import (
            build_shogi_policy_value_tensor_cache_shard,
        )

        remote_bundle = str(task["remote_bundle"])
        cache_name = str(task["cache_name"])
        data_selection_path = VOLUME_ROOT / remote_bundle / "data-selection.json"
        cache_dir = VOLUME_ROOT / remote_bundle / "cache" / cache_name
        result = build_shogi_policy_value_tensor_cache_shard(
            data_selection_path=data_selection_path,
            cache_dir=cache_dir,
            split=str(task["split"]),
            source_index=int(task["source_index"]),
            source_example_start_index=int(task["source_example_start_index"]),
            source_example_end_index=int(task["source_example_end_index"]),
            shard_index=int(task["shard_index"]),
            resume=True,
            output_space=str(task["output_space"]),
        )
        volume.commit()
        return result

    @app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=60 * 60)
    def reset_remote_cache(remote_bundle: str, cache_name: str) -> dict[str, object]:
        cache_dir = VOLUME_ROOT / remote_bundle / "cache" / cache_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        volume.commit()
        return {"remote_cache": f"{remote_bundle}/cache/{cache_name}", "removed": True}

    @app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=60 * 60)
    def write_remote_manifest(remote_bundle: str, shard_examples: int, cache_name: str, output_space: str) -> dict[str, object]:
        from intrep.problems.shogi_policy_value.tensor_cache import (
            write_shogi_policy_value_tensor_cache_manifest,
        )

        data_selection_path = VOLUME_ROOT / remote_bundle / "data-selection.json"
        cache_dir = VOLUME_ROOT / remote_bundle / "cache" / cache_name
        result = write_shogi_policy_value_tensor_cache_manifest(
            data_selection_path=data_selection_path,
            cache_dir=cache_dir,
            shard_examples=shard_examples,
            output_space=output_space,
        )
        volume.commit()
        return result

    @app.local_entrypoint()
    def main(
        local_bundle: str = str(DEFAULT_LOCAL_BUNDLE),
        remote_bundle: str = DEFAULT_REMOTE_BUNDLE,
        shard_examples: int = 100_000,
        split: str = "all",
        skip_upload: bool = False,
        limit_shards: int = 0,
        reset_cache: bool = False,
        output_space: str = DEFAULT_OUTPUT_SPACE,
    ) -> None:
        run(
            local_bundle=Path(local_bundle),
            remote_bundle=remote_bundle,
            shard_examples=shard_examples,
            split=split,
            skip_upload=skip_upload,
            limit_shards=limit_shards or None,
            reset_cache=reset_cache,
            output_space=output_space,
        )

else:
    app = None


def run(
    *,
    local_bundle: Path,
    remote_bundle: str,
    shard_examples: int,
    split: str,
    skip_upload: bool,
    limit_shards: int | None,
    reset_cache: bool,
    output_space: str,
) -> None:
    if modal is None:
        raise RuntimeError("Modal is required. Run with: uv run --with modal modal run scripts/modal_build_shogi_policy_value_tensor_cache.py")
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    if split not in {"all", "train", "eval"}:
        raise ValueError("split must be all, train, or eval")
    if limit_shards is not None and limit_shards <= 0:
        raise ValueError("limit_shards must be positive")
    if output_space not in {"candidate_move", "policy_plane"}:
        raise ValueError("output_space must be candidate_move or policy_plane")

    local_bundle = local_bundle.resolve()
    cache_name = _cache_name_for_output_space(output_space)
    data_selection_path = local_bundle / "data-selection.json"
    if not data_selection_path.exists():
        raise FileNotFoundError(data_selection_path)

    if not skip_upload:
        _upload_bundle(local_bundle=local_bundle, remote_bundle=remote_bundle)

    if reset_cache:
        print(json.dumps(reset_remote_cache.remote(remote_bundle, cache_name), indent=2))

    tasks = _build_tasks(
        local_data_selection_path=data_selection_path,
        remote_bundle=remote_bundle,
        shard_examples=shard_examples,
        split=split,
        output_space=output_space,
        cache_name=cache_name,
    )
    if limit_shards is not None:
        tasks = tasks[:limit_shards]

    print(json.dumps({"volume": VOLUME_NAME, "remote_bundle": remote_bundle, "shard_count": len(tasks)}, indent=2))
    results = list(build_remote_shard.map(tasks))
    manifest = write_remote_manifest.remote(remote_bundle, shard_examples, cache_name, output_space)
    print(
        json.dumps(
            {
                "volume": VOLUME_NAME,
                "remote_cache": f"{remote_bundle}/cache/{cache_name}",
                "output_space": output_space,
                "built_shards": len(results),
                "train_count": manifest["train_count"],
                "eval_count": manifest["eval_count"],
                "skipped_example_count": manifest["skipped_example_count"],
                "shard_count": len(manifest["shards"]),
            },
            indent=2,
        )
    )


def _upload_bundle(*, local_bundle: Path, remote_bundle: str) -> None:
    remote_path = f"/{remote_bundle}"
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_bundle), remote_path)


def _build_tasks(
    *,
    local_data_selection_path: Path,
    remote_bundle: str,
    shard_examples: int,
    split: str,
    output_space: str,
    cache_name: str,
) -> list[dict[str, object]]:
    payload = json.loads(local_data_selection_path.read_text(encoding="utf-8"))
    local_bundle = local_data_selection_path.parent
    tasks: list[dict[str, object]] = []
    split_names = ("train", "eval") if split == "all" else (split,)
    for split_name in split_names:
        sources = _object_list(payload[f"{split_name}_sources"])
        for source_index, source in enumerate(sources):
            source_payload = _object_dict(source)
            source_path = _local_source_path(local_bundle, source_payload)
            example_count = _count_jsonl_records(source_path)
            if "max_examples" in source_payload:
                example_count = min(example_count, int(source_payload["max_examples"]))
            shard_index = 0
            for start in range(0, example_count, shard_examples):
                end = min(start + shard_examples, example_count)
                tasks.append(
                    {
                        "remote_bundle": remote_bundle,
                        "cache_name": cache_name,
                        "output_space": output_space,
                        "split": split_name,
                        "source_index": source_index,
                        "source_example_start_index": start,
                        "source_example_end_index": end,
                        "shard_index": shard_index,
                    }
                )
                shard_index += 1
    return tasks


def _cache_name_for_output_space(output_space: str) -> str:
    if output_space == "policy_plane":
        return DEFAULT_POLICY_PLANE_CACHE_NAME
    return DEFAULT_CACHE_NAME


def _local_source_path(local_bundle: Path, source: dict[str, object]) -> Path:
    path = Path(str(source["path"]))
    if path.is_absolute():
        return path
    return local_bundle / path


def _count_jsonl_records(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("expected object")
    return value


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return value


if __name__ == "__main__":
    if modal is None:
        raise SystemExit("Run with: uv run --with modal modal run scripts/modal_build_shogi_policy_value_tensor_cache.py")
