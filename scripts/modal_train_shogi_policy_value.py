from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - Modal is supplied by `modal run`.
    modal = None  # type: ignore[assignment]


APP_NAME = "intrep-shogi-policy-value-training"
VOLUME_NAME = os.environ.get("INTREP_MODAL_VOLUME_NAME", "intrep-shogi-tensor-cache")
GPU_TYPE = os.environ.get("INTREP_MODAL_GPU", "L4")
VOLUME_ROOT = Path("/data")
DEFAULT_REMOTE_BUNDLE = "qhapaq-full"
DEFAULT_CACHE_NAME = "shogi-policy-value-tensors"
DEFAULT_LOCAL_INIT_CHECKPOINT = Path("models/d256-h1024-heads8-l6-shogi/checkpoint.pt")
DEFAULT_REMOTE_INIT_CHECKPOINT = "d256-h1024-heads8-l6-shogi"


if modal is not None:
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy>=1.26", "python-shogi>=1.1.1", "tokenizers>=0.23.1", "torch>=2.2")
        .env({"PYTHONPATH": "/root/src"})
        .add_local_dir("src", "/root/src")
    )
    app = modal.App(APP_NAME)

    @app.function(image=image, gpu=GPU_TYPE, volumes={str(VOLUME_ROOT): volume}, timeout=24 * 60 * 60)
    def train_remote(config: dict[str, object]) -> dict[str, object]:
        import torch

        from intrep.train_shogi_policy_value import main as train_main

        remote_bundle = str(config["remote_bundle"])
        run_name = str(config["run_name"])
        batch_size = int(config["batch_size"])
        max_steps = int(config["max_steps"])
        if max_steps <= 0:
            manifest_path = VOLUME_ROOT / remote_bundle / "cache" / DEFAULT_CACHE_NAME / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            max_steps = math.ceil(int(manifest["train_count"]) / batch_size)

        output_dir = VOLUME_ROOT / remote_bundle / "training-runs" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        data_selection_path = VOLUME_ROOT / remote_bundle / "data-selection.json"
        tensor_cache_path = VOLUME_ROOT / remote_bundle / "cache" / DEFAULT_CACHE_NAME
        init_checkpoint_path = VOLUME_ROOT / remote_bundle / "init-checkpoints" / str(config["remote_init_checkpoint"]) / "checkpoint.pt"
        timing_path = output_dir / "timing.json"

        argv = [
            "train_shogi_policy_value",
            "--data-selection",
            str(data_selection_path),
            "--tensor-cache",
            str(tensor_cache_path),
            "--init-checkpoint-path",
            str(init_checkpoint_path),
            "--checkpoint-path",
            str(output_dir / "checkpoint.pt"),
            "--best-checkpoint-path",
            str(output_dir / "best-checkpoint.pt"),
            "--metrics-path",
            str(output_dir / "metrics.json"),
            "--max-steps",
            str(max_steps),
            "--batch-size",
            str(batch_size),
            "--learning-rate",
            str(config["learning_rate"]),
            "--weight-decay",
            str(config["weight_decay"]),
            "--embedding-dim",
            str(config["embedding_dim"]),
            "--hidden-dim",
            str(config["hidden_dim"]),
            "--num-heads",
            str(config["num_heads"]),
            "--num-layers",
            str(config["num_layers"]),
            "--policy-loss-weight",
            str(config["policy_loss_weight"]),
            "--value-loss-weight",
            str(config["value_loss_weight"]),
            "--device",
            "cuda",
            "--log-every",
            str(config["log_every"]),
            "--num-workers",
            str(config["num_workers"]),
            "--checkpoint-every",
            str(config["checkpoint_every"]),
            "--metrics-every",
            str(config["metrics_every"]),
            "--keep-last-n-checkpoints",
            str(config["keep_last_n_checkpoints"]),
            "--pin-memory",
        ]
        if int(config["max_train_eval_examples"]) > 0:
            argv.extend(["--max-train-eval-examples", str(config["max_train_eval_examples"])])
        if int(config["max_eval_examples"]) > 0:
            argv.extend(["--max-eval-examples", str(config["max_eval_examples"])])
        if int(config["eval_every"]) > 0:
            argv.extend(["--eval-every", str(config["eval_every"])])

        started = time.monotonic()
        import sys

        old_argv = sys.argv
        sys.argv = argv
        try:
            train_main()
        finally:
            sys.argv = old_argv
        elapsed_seconds = time.monotonic() - started
        metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        timing = {
            "schema_version": "intrep.shogi_policy_value_training_timing.v1",
            "elapsed_seconds": elapsed_seconds,
            "gpu_type": GPU_TYPE,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "run_name": run_name,
            "remote_bundle": remote_bundle,
            "batch_size": batch_size,
            "max_steps": max_steps,
            "raw_train_case_count": metrics["raw_train_case_count"],
            "raw_eval_case_count": metrics["raw_eval_case_count"],
            "actual_steps": metrics["metrics"]["actual_steps"],
            "steps_per_second": metrics["metrics"]["actual_steps"] / elapsed_seconds if elapsed_seconds else 0.0,
            "initial_eval_loss": metrics["metrics"]["initial_eval_loss"],
            "eval_loss": metrics["metrics"]["eval_loss"],
            "initial_eval_accuracy": metrics["metrics"]["initial_eval_accuracy"],
            "eval_accuracy": metrics["metrics"]["eval_accuracy"],
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
        volume.commit()
        return {
            "volume": VOLUME_NAME,
            "output_dir": str(output_dir.relative_to(VOLUME_ROOT)),
            **timing,
        }

    @app.local_entrypoint()
    def main(
        remote_bundle: str = DEFAULT_REMOTE_BUNDLE,
        run_name: str = "qhapaq-full-d256-one-epoch-001",
        local_init_checkpoint: str = str(DEFAULT_LOCAL_INIT_CHECKPOINT),
        remote_init_checkpoint: str = DEFAULT_REMOTE_INIT_CHECKPOINT,
        skip_upload_init_checkpoint: bool = False,
        max_steps: int = 0,
        batch_size: int = 512,
        learning_rate: float = 0.0005,
        weight_decay: float = 0.01,
        embedding_dim: int = 256,
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 6,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 0.0,
        num_workers: int = 2,
        log_every: int = 100,
        checkpoint_every: int = 1000,
        metrics_every: int = 1000,
        keep_last_n_checkpoints: int = 3,
        eval_every: int = 0,
        max_train_eval_examples: int = 0,
        max_eval_examples: int = 0,
    ) -> None:
        if not skip_upload_init_checkpoint:
            _upload_init_checkpoint(
                local_checkpoint=Path(local_init_checkpoint),
                remote_bundle=remote_bundle,
                remote_init_checkpoint=remote_init_checkpoint,
            )
        result = train_remote.remote(
            {
                "remote_bundle": remote_bundle,
                "run_name": run_name,
                "remote_init_checkpoint": remote_init_checkpoint,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "policy_loss_weight": policy_loss_weight,
                "value_loss_weight": value_loss_weight,
                "num_workers": num_workers,
                "log_every": log_every,
                "checkpoint_every": checkpoint_every,
                "metrics_every": metrics_every,
                "keep_last_n_checkpoints": keep_last_n_checkpoints,
                "eval_every": eval_every,
                "max_train_eval_examples": max_train_eval_examples,
                "max_eval_examples": max_eval_examples,
            }
        )
        print(json.dumps(result, indent=2))

else:
    app = None


def _upload_init_checkpoint(*, local_checkpoint: Path, remote_bundle: str, remote_init_checkpoint: str) -> None:
    if modal is None:
        raise RuntimeError("Modal is required. Run with: uv run --with modal modal run scripts/modal_train_shogi_policy_value.py")
    local_checkpoint = local_checkpoint.resolve()
    if not local_checkpoint.exists():
        raise FileNotFoundError(local_checkpoint)
    with volume.batch_upload() as batch:
        batch.put_directory(str(local_checkpoint.parent), f"/{remote_bundle}/init-checkpoints/{remote_init_checkpoint}")


if __name__ == "__main__":
    if modal is None:
        raise SystemExit("Run with: uv run --with modal modal run scripts/modal_train_shogi_policy_value.py")
