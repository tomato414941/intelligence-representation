from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from intrep.shogi_game_record import load_shogi_move_choice_examples_from_game_records_jsonl
from intrep.shogi_move_choice import (
    load_shogi_move_choice_examples_jsonl,
    write_shogi_move_choice_examples_jsonl,
)
from intrep.shogi_move_choice_checkpoint import save_shogi_move_choice_checkpoint
from intrep.shogi_move_choice_training import ShogiMoveChoiceTrainingConfig, train_shogi_move_choice_model


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
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train-eval-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
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
        value_loss_weight=args.value_loss_weight,
        device=args.device,
        max_train_eval_examples=args.max_train_eval_examples,
        max_eval_examples=args.max_eval_examples,
    )
    result = train_shogi_move_choice_model(train_examples, eval_examples=eval_examples, config=config)
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    save_shogi_move_choice_checkpoint(args.checkpoint_path, result)
    metrics = {
        "train_case_count": len(train_examples),
        "eval_case_count": len(eval_examples) if eval_examples is not None else 0,
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


if __name__ == "__main__":
    main()
