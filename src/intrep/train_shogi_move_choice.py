from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.shogi.game_record import load_shogi_move_choice_examples_from_game_records_jsonl
from intrep.shogi.move_choice import (
    load_shogi_move_choice_examples_jsonl,
    write_shogi_move_choice_examples_jsonl,
)
from intrep.shogi.move_choice_checkpoint import (
    save_shogi_move_choice_checkpoint,
    save_shogi_move_choice_model_checkpoint,
)
from intrep.shogi.move_choice_training import (
    ShogiMoveChoiceTrainingConfig,
    ShogiMoveChoiceTrainingProgress,
    train_shogi_move_choice_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a shogi move-choice policy/value model.")
    parser.add_argument("--train-games-jsonl", type=Path)
    parser.add_argument("--eval-games-jsonl", type=Path)
    parser.add_argument("--train-examples-jsonl", type=Path)
    parser.add_argument("--eval-examples-jsonl", type=Path)
    parser.add_argument("--write-train-examples-jsonl", type=Path)
    parser.add_argument("--write-eval-examples-jsonl", type=Path)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-eval-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--log-every", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--metrics-every", type=int)
    parser.add_argument("--keep-last-n-checkpoints", type=int)
    args = parser.parse_args()

    train_examples = _load_examples(args.train_examples_jsonl, args.train_games_jsonl)
    eval_examples = _load_examples(args.eval_examples_jsonl, args.eval_games_jsonl) if (
        args.eval_examples_jsonl is not None or args.eval_games_jsonl is not None
    ) else None
    if args.write_train_examples_jsonl is not None:
        write_shogi_move_choice_examples_jsonl(args.write_train_examples_jsonl, train_examples)
    if args.write_eval_examples_jsonl is not None and eval_examples is not None:
        write_shogi_move_choice_examples_jsonl(args.write_eval_examples_jsonl, eval_examples)

    config = ShogiMoveChoiceTrainingConfig(
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        progress_every=_progress_every(args.checkpoint_every, args.metrics_every),
    )
    # Periodic artifacts keep long disposable-pod runs from losing all progress
    # when the process ends before final checkpoint and metrics are written.
    progress_writer = _ProgressArtifactWriter(args)
    result = train_shogi_move_choice_model(
        train_examples,
        eval_examples=eval_examples,
        config=config,
        progress_callback=progress_writer.write,
    )
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_shogi_move_choice_checkpoint(args.checkpoint_path, result)
    metrics = {
        "raw_train_case_count": len(train_examples),
        "raw_eval_case_count": len(eval_examples) if eval_examples is not None else 0,
        "used_eval_case_count": result.metrics.eval_case_count,
        "checkpoint_path": str(args.checkpoint_path),
        "config": asdict(result.config),
        "metrics": asdict(result.metrics),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _load_examples(examples_jsonl: Path | None, games_jsonl: Path | None):
    if examples_jsonl is not None:
        return load_shogi_move_choice_examples_jsonl(examples_jsonl)
    if games_jsonl is not None:
        return load_shogi_move_choice_examples_from_game_records_jsonl(games_jsonl)
    raise ValueError("either examples jsonl or games jsonl must be provided")


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

    def write(self, progress: ShogiMoveChoiceTrainingProgress) -> None:
        if self.checkpoint_every is not None and progress.step % self.checkpoint_every == 0:
            self._write_checkpoint(progress)
        if self.metrics_every is not None and progress.step % self.metrics_every == 0:
            self._write_metrics(progress)

    def _write_checkpoint(self, progress: ShogiMoveChoiceTrainingProgress) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / f"checkpoint_step_{progress.step}.pt"
        save_shogi_move_choice_model_checkpoint(path, progress.model, progress.config)
        self.saved_checkpoints.append(path)
        self._prune_checkpoints()

    def _write_metrics(self, progress: ShogiMoveChoiceTrainingProgress) -> None:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        path = self.metrics_dir / f"metrics_step_{progress.step}.json"
        payload = {
            "step": progress.step,
            "max_steps": progress.max_steps,
            "loss": progress.loss,
            "elapsed_seconds": progress.elapsed_seconds,
            "config": asdict(progress.config),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _prune_checkpoints(self) -> None:
        if self.keep_last_n_checkpoints is None:
            return
        while len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
            path = self.saved_checkpoints.pop(0)
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
