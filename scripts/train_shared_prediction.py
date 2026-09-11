"""Train every declared data source against one LFM body with exchangeable heads."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from intrep.problems.shared_prediction.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    initialize = parser.add_mutually_exclusive_group(required=True)
    initialize.add_argument("--base", type=Path)
    initialize.add_argument("--resume", type=Path)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True, help="total joint updates, including restored updates")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--extend", action="store_true", help="retain old sources and optimizer state while adding sources/heads")
    parser.add_argument("--audit-gradients", action="store_true")
    parser.add_argument("--evaluation-examples", type=int, default=1, help="fixed cases per source; native uses worlds with all transitions")
    parser.add_argument("--evaluation-interval", type=int, default=0)
    parser.add_argument("--native-controls", action="store_true", help="evaluate omission of individual native input forms")
    parser.add_argument("--prompts", type=Path, help="JSON list of fixed generation prompts and optional expected answers")
    parser.add_argument("--extension", action="append", default=[], help="explicit Python module registering additional source factories")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    for module in args.extension:
        importlib.import_module(module)
    recipe = json.loads(args.recipe.read_text())
    train(base=args.base, resume=args.resume, recipe=recipe, root=args.data_root.resolve(),
          output=args.output, steps=args.steps, device=args.device, optimizer=args.optimizer,
          learning_rate=args.learning_rate, max_grad_norm=args.max_grad_norm, extend=args.extend,
          audit_gradients=args.audit_gradients, extensions=args.extension, checkpoint_interval=args.checkpoint_interval,
          evaluation_examples=args.evaluation_examples, evaluation_interval=args.evaluation_interval,
          native_controls=args.native_controls, prompts=json.loads(args.prompts.read_text()) if args.prompts else None)


if __name__ == "__main__":
    main()
