from __future__ import annotations

# Keep this as a dataset-specific cache recipe, not a general job framework.
# Modal is used to avoid spending local CPU on full-cache regeneration.

import subprocess
import os
from pathlib import Path

import modal


APP_ROOT = "/workspace/intelligence-representation"
GAMES_JSONL = f"{APP_ROOT}/data/qhapaq/processed/qhapaq_all_games.jsonl"
VOLUME_NAME = "intrep-shogi-cache"

app = modal.App("intrep-shogi-example-cache")
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("python-shogi>=1.1.1")
    .add_local_dir("src", remote_path=f"{APP_ROOT}/src")
    .add_local_file(
        "data/qhapaq/processed/qhapaq_all_games.jsonl",
        remote_path=GAMES_JSONL,
    )
)


@app.function(
    image=image,
    volumes={"/cache": cache_volume},
    cpu=1.0,
    memory=2048,
    timeout=4 * 60 * 60,
)
def prepare_shard(shard_index: int, shard_count: int, max_games: int | None = None) -> dict[str, object]:
    output_path = f"/cache/shogi/qhapaq-all-move-choice-examples.part-{shard_index:05d}-of-{shard_count:05d}.jsonl"
    failure_path = (
        f"/cache/shogi/qhapaq-all-move-choice-examples.failures.part-{shard_index:05d}-of-{shard_count:05d}.jsonl"
    )
    command = [
        "python",
        "-m",
        "intrep.prepare_shogi_move_choice_examples",
        "--games-jsonl",
        GAMES_JSONL,
        "--examples-jsonl",
        output_path,
        "--failures-jsonl",
        failure_path,
        "--shard-index",
        str(shard_index),
        "--shard-count",
        str(shard_count),
        "--progress-every",
        "1000",
    ]
    if max_games is not None:
        command.extend(["--max-games", str(max_games)])
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{APP_ROOT}/src"
    completed = subprocess.run(
        command,
        cwd=APP_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"prepare shard failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    cache_volume.commit()
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "output_path": output_path,
        "failure_path": failure_path,
        "stdout": completed.stdout,
    }


@app.local_entrypoint()
def main(shard_count: int = 8, start: int = 0, end: int | None = None, max_games: int | None = None) -> None:
    end = shard_count if end is None else end
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= start <= end <= shard_count:
        raise ValueError("expected 0 <= start <= end <= shard_count")
    for result in prepare_shard.starmap(
        [(index, shard_count, max_games) for index in range(start, end)],
        order_outputs=False,
    ):
        print(result)
