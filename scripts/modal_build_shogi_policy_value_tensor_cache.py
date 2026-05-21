from __future__ import annotations

"""Build shogi policy/value tensor caches on Modal CPU workers.

The durable artifact is the released local tensor-cache directory. The Modal
Volume is the remote build workspace, not the training input of record.
"""

import json
import os
import shutil
import subprocess
import threading
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
DEFAULT_CACHE_NAME = "legal-move"
DEFAULT_ACTION_PLANE_POLICY_CACHE_NAME = "action-plane-policy"
DEFAULT_OUTPUT_SPACE = "legal_move"
DEFAULT_RELEASE = "local"
DEFAULT_SHARD_EXAMPLES = 10_000
WORKER_MEMORY_MB = int(os.environ.get("INTREP_MODAL_TENSOR_CACHE_WORKER_MEMORY_MB", "8192"))
PROGRESS_INTERVAL_SECONDS = float(os.environ.get("INTREP_MODAL_TENSOR_CACHE_PROGRESS_SECONDS", "30"))


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy>=1.26", "python-shogi>=1.1.1", "tokenizers>=0.23.1", "torch>=2.2")
        .env({"PYTHONPATH": "/root/src"})
        .add_local_dir("src", "/root/src")
    )
    app = modal.App(APP_NAME)

    @app.function(image=image, volumes={str(VOLUME_ROOT): volume}, timeout=24 * 60 * 60, memory=WORKER_MEMORY_MB)
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
        cache_dir.mkdir(parents=True, exist_ok=True)
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
        shard_examples: int = DEFAULT_SHARD_EXAMPLES,
        split: str = "all",
        skip_upload: bool = False,
        limit_shards: int = 0,
        reset_cache: bool = False,
        output_space: str = DEFAULT_OUTPUT_SPACE,
        release: str = DEFAULT_RELEASE,
        local_cache: str = "",
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
            release=release,
            local_cache=Path(local_cache) if local_cache else None,
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
    release: str,
    local_cache: Path | None,
) -> None:
    if modal is None:
        raise RuntimeError("Modal is required. Run with: uv run --with modal modal run scripts/modal_build_shogi_policy_value_tensor_cache.py")
    if shard_examples <= 0:
        raise ValueError("shard_examples must be positive")
    if split not in {"all", "train", "eval"}:
        raise ValueError("split must be all, train, or eval")
    if limit_shards is not None and limit_shards <= 0:
        raise ValueError("limit_shards must be positive")
    if output_space not in {"legal_move", "action_plane_policy"}:
        raise ValueError("output_space must be legal_move or action_plane_policy")
    if release not in {"local", "volume"}:
        raise ValueError("release must be local or volume")

    local_bundle = local_bundle.resolve()
    cache_name = _cache_name_for_output_space(output_space)
    local_cache_path = _local_cache_path(
        local_bundle=local_bundle,
        output_space=output_space,
        local_cache=local_cache,
    )
    data_selection_path = local_bundle / "data-selection.json"
    if not data_selection_path.exists():
        raise FileNotFoundError(data_selection_path)

    if not skip_upload:
        _log_event("upload_start", local_bundle=str(local_bundle), remote_bundle=remote_bundle)
        _upload_bundle(local_bundle=local_bundle, remote_bundle=remote_bundle)
        _log_event("upload_done", remote_bundle=remote_bundle)

    if reset_cache:
        _log_event("reset_start", remote_bundle=remote_bundle, cache_name=cache_name)
        reset_result = reset_remote_cache.remote(remote_bundle, cache_name)
        _log_event("reset_done", **reset_result)

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

    _log_event("build_start", volume=VOLUME_NAME, remote_bundle=remote_bundle, cache_name=cache_name, shard_count=len(tasks))
    monitor = _RemoteShardProgressMonitor(
        remote_bundle=remote_bundle,
        cache_name=cache_name,
        tasks=tasks,
        interval_seconds=PROGRESS_INTERVAL_SECONDS,
    )
    monitor.start()
    try:
        results = list(build_remote_shard.map(tasks))
    finally:
        monitor.stop()
    _log_event("build_done", built_shards=len(results), shard_count=len(tasks))

    _log_event("manifest_start", remote_bundle=remote_bundle, cache_name=cache_name)
    manifest = write_remote_manifest.remote(remote_bundle, shard_examples, cache_name, output_space)
    _log_event(
        "manifest_done",
        train_count=manifest["train_count"],
        eval_count=manifest["eval_count"],
        skipped_example_count=manifest["skipped_example_count"],
        shard_count=len(manifest["shards"]),
    )
    remote_cache = f"{remote_bundle}/cache/{cache_name}"
    release_result: dict[str, object] | None = None
    if release == "local":
        _log_event("release_start", remote_cache=remote_cache, local_cache=str(local_cache_path))
        release_result = _release_remote_cache_to_local(
            remote_bundle=remote_bundle,
            cache_name=cache_name,
            local_cache=local_cache_path,
        )
        _log_event("release_done", **release_result)

    _log_event(
        "complete",
        volume=VOLUME_NAME,
        remote_cache=remote_cache,
        local_cache=str(local_cache_path) if release == "local" else None,
        release=release,
        output_space=output_space,
        built_shards=len(results),
        train_count=manifest["train_count"],
        eval_count=manifest["eval_count"],
        skipped_example_count=manifest["skipped_example_count"],
        shard_count=len(manifest["shards"]),
        release_result=release_result,
    )


def _upload_bundle(*, local_bundle: Path, remote_bundle: str) -> None:
    remote_path = f"/{remote_bundle}"
    with volume.batch_upload(force=True) as batch:
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


class _RemoteShardProgressMonitor:
    def __init__(
        self,
        *,
        remote_bundle: str,
        cache_name: str,
        tasks: list[dict[str, object]],
        interval_seconds: float,
    ) -> None:
        self.remote_bundle = remote_bundle
        self.cache_name = cache_name
        self.expected_paths = {_task_shard_manifest_relative_path(task) for task in tasks}
        self.interval_seconds = max(1.0, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_completed_count = -1

    def start(self) -> None:
        if not self.expected_paths:
            _log_event("shard_progress", completed_shards=0, shard_count=0)
            return
        self._thread = threading.Thread(target=self._run, name="modal-tensor-cache-progress", daemon=True)
        self._thread.start()
        self.report()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.report(force=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.report()

    def report(self, *, force: bool = False) -> None:
        try:
            completed_paths = _completed_shard_manifest_paths(
                remote_bundle=self.remote_bundle,
                cache_name=self.cache_name,
            )
        except Exception as exc:  # noqa: BLE001
            _log_event("shard_progress_error", error=str(exc))
            return

        completed_count = len(self.expected_paths & completed_paths)
        if not force and completed_count == self._last_completed_count:
            return
        self._last_completed_count = completed_count
        _log_event(
            "shard_progress",
            completed_shards=completed_count,
            shard_count=len(self.expected_paths),
            remaining_shards=len(self.expected_paths) - completed_count,
        )


def _completed_shard_manifest_paths(*, remote_bundle: str, cache_name: str) -> set[str]:
    cache_prefix = f"{remote_bundle}/cache/{cache_name}/"
    entries = volume.listdir(f"/{remote_bundle}/cache/{cache_name}", recursive=True)
    paths: set[str] = set()
    for entry in entries:
        path = str(entry.path)
        if not path.endswith(".json"):
            continue
        relative_path = path.removeprefix(cache_prefix)
        if relative_path == "manifest.json":
            continue
        paths.add(relative_path)
    return paths


def _task_shard_manifest_relative_path(task: dict[str, object]) -> str:
    split = str(task["split"])
    source_index = int(task["source_index"])
    start = int(task["source_example_start_index"])
    end = int(task["source_example_end_index"])
    return f"{split}/source-{source_index:04d}-examples-{start:08d}-{end:08d}.json"


def _log_event(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _cache_name_for_output_space(output_space: str) -> str:
    if output_space == "action_plane_policy":
        return DEFAULT_ACTION_PLANE_POLICY_CACHE_NAME
    return DEFAULT_CACHE_NAME


def _local_cache_path(*, local_bundle: Path, output_space: str, local_cache: Path | None) -> Path:
    if local_cache is not None:
        return local_cache.resolve()
    return local_bundle / "cache" / _cache_name_for_output_space(output_space)


def _release_remote_cache_to_local(
    *,
    remote_bundle: str,
    cache_name: str,
    local_cache: Path,
) -> dict[str, object]:
    remote_cache = f"/{remote_bundle}/cache/{cache_name}"
    if local_cache.exists():
        if local_cache.is_dir():
            shutil.rmtree(local_cache)
        else:
            local_cache.unlink()
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "--force",
            VOLUME_NAME,
            remote_cache,
            str(local_cache.parent),
        ],
        check=True,
    )
    return {
        "volume": VOLUME_NAME,
        "remote_cache": remote_cache,
        "local_cache": str(local_cache),
    }


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
