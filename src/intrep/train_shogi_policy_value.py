from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.tasks.shogi_policy_value.checkpoint import (
    load_shogi_policy_value_checkpoint_state_dict,
    save_shogi_policy_value_checkpoint,
    save_shogi_policy_value_model_checkpoint,
    save_shogi_policy_value_state_checkpoint,
)
from intrep.tasks.shogi_policy_value.dataset_definition import (
    load_shogi_policy_value_dataset_definition,
    load_shogi_policy_value_dataset_examples,
    shogi_policy_value_dataset_definition_to_json,
)
from intrep.tasks.shogi_policy_value.examples import ShogiPolicyValueExample
from intrep.tasks.shogi_policy_value.training import (
    ShogiPolicyValueTrainingConfig,
    ShogiPolicyValueTrainingProgress,
    train_shogi_policy_value_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shogi policy/value model.")
    parser.add_argument("--dataset-definition", type=Path, required=True)
    parser.add_argument("--init-checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--best-checkpoint-path", type=Path)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-eval-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--metrics-every", type=int)
    parser.add_argument("--keep-last-n-checkpoints", type=int)
    args = parser.parse_args()

    dataset_definition = load_shogi_policy_value_dataset_definition(args.dataset_definition)
    train_examples, eval_examples = load_shogi_policy_value_dataset_examples(dataset_definition)

    config = ShogiPolicyValueTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_shared_core=True,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        device=args.device,
        max_train_eval_examples=args.max_train_eval_examples,
        max_eval_examples=args.max_eval_examples,
        log_every=args.log_every,
        eval_every=args.eval_every,
        early_stopping_patience=args.early_stopping_patience,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        progress_every=_progress_every(args.checkpoint_every, args.metrics_every),
    )
    # Periodic artifacts keep long disposable-pod runs from losing all progress
    # when the process ends before final checkpoint and metrics are written.
    progress_writer = _ProgressArtifactWriter(args)
    result = train_shogi_policy_value_model(
        train_examples,
        eval_examples=eval_examples,
        config=config,
        initial_state_dict=(
            load_shogi_policy_value_checkpoint_state_dict(args.init_checkpoint_path, device=args.device)
            if args.init_checkpoint_path is not None
            else None
        ),
        progress_callback=progress_writer.write,
    )
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_shogi_policy_value_checkpoint(args.checkpoint_path, result)
    if args.best_checkpoint_path is not None and result.best_model_state_dict is not None:
        args.best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_shogi_policy_value_state_checkpoint(args.best_checkpoint_path, result.best_model_state_dict, result.config)
    metrics = {
        "raw_train_case_count": len(train_examples),
        "raw_eval_case_count": len(eval_examples),
        "used_eval_case_count": result.metrics.eval_case_count,
        "train_policy_target_summary": _policy_target_summary(train_examples),
        "eval_policy_target_summary": _policy_target_summary(eval_examples),
        "dataset_definition_path": str(args.dataset_definition),
        "dataset_definition": shogi_policy_value_dataset_definition_to_json(dataset_definition),
        "init_checkpoint_path": str(args.init_checkpoint_path) if args.init_checkpoint_path is not None else None,
        "checkpoint_path": str(args.checkpoint_path),
        "best_checkpoint_path": str(args.best_checkpoint_path) if args.best_checkpoint_path is not None else None,
        "config": asdict(result.config),
        "metrics": asdict(result.metrics),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _policy_target_summary(examples: list[ShogiPolicyValueExample]) -> dict[str, float | int]:
    available_counts = [
        sum(1 for weight in example.policy_targets.values() if weight > 0.0)
        for example in examples
        if example.policy_targets is not None
    ]
    available_count = len(available_counts)
    total_count = len(examples)
    return {
        "available_count": available_count,
        "missing_count": total_count - available_count,
        "available_ratio": available_count / total_count if total_count else 0.0,
        "mean_nonzero_count": sum(available_counts) / available_count if available_count else 0.0,
    }


def _progress_every(*values: int | None) -> int | None:
    intervals = [value for value in values if value is not None]
    return min(intervals) if intervals else None


class _ProgressArtifactWriter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.checkpoint_every = args.checkpoint_every
        self.metrics_every = args.metrics_every
        self.keep_last_n_checkpoints = args.keep_last_n_checkpoints
        self.checkpoint_dir = args.checkpoint_path.parent
        self.metrics_dir = args.metrics_path.parent
        self.saved_checkpoints: list[Path] = []
        if self.checkpoint_every is not None and self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.metrics_every is not None and self.metrics_every <= 0:
            raise ValueError("metrics_every must be positive")
        if self.keep_last_n_checkpoints is not None and self.keep_last_n_checkpoints <= 0:
            raise ValueError("keep_last_n_checkpoints must be positive")

    def write(self, progress: ShogiPolicyValueTrainingProgress) -> None:
        if self.checkpoint_every is not None and progress.step % self.checkpoint_every == 0:
            self._write_checkpoint(progress)
        if self.metrics_every is not None and progress.step % self.metrics_every == 0:
            self._write_metrics(progress)

    def _write_checkpoint(self, progress: ShogiPolicyValueTrainingProgress) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_step_{progress.step}.pt"
        save_shogi_policy_value_model_checkpoint(path, progress.model, progress.config)
        self.saved_checkpoints.append(path)
        self._prune_checkpoints()

    def _write_metrics(self, progress: ShogiPolicyValueTrainingProgress) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"metrics_step_{progress.step}.json"
        payload = {
            "step": progress.step,
            "max_steps": progress.max_steps,
            "loss": progress.loss,
            "elapsed_seconds": progress.elapsed_seconds,
            "config": asdict(progress.config),
        }
        if progress.eval_metrics is not None:
            payload["eval_metrics"] = asdict(progress.eval_metrics)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _prune_checkpoints(self) -> None:
        if self.keep_last_n_checkpoints is None:
            return
        while len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
            path = self.saved_checkpoints.pop(0)
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
