"""Measure direct and mediated order transfer on one fixed development/holdout split."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from intrep.problems.shared_prediction.rule_transfer import MeasuredReadout, evaluate_transfer
from intrep.problems.shared_prediction.rule_transfer_data import file_digest, load_panel
from intrep.problems.shared_prediction.sources import Source
from intrep.problems.shared_prediction.training import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--order", choices=("a", "b"), required=True)
    parser.add_argument("--extension", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("use a new output file and a positive CPU thread count")
    torch.set_num_threads(args.threads)
    for extension in args.extension:
        importlib.import_module(extension)
    artifact_stats = {path: path.stat() for path in (args.panel, args.checkpoint)}
    panel_sha256, checkpoint_sha256 = file_digest(args.panel), file_digest(args.checkpoint)

    def check_artifacts():
        for path, before in artifact_stats.items():
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("a fixed panel/checkpoint changed during evaluation")

    panel, images, labels = load_panel(args.panel, args.data_root)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=args.device, extensions=args.extension)
    completed_steps = payload["trainer"]["steps"]
    del payload
    check_artifacts()
    if "rgb" not in model.input_heads or "text" not in model.output_heads:
        parser.error("the shared checkpoint must contain RGB input and language output layers")
    source = Source(model, tokenizer, {}, args.data_root)
    result = evaluate_transfer(source, panel, images, labels, split=args.split, order_name=args.order,
                               readout=MeasuredReadout(source, args.max_tokens),
                               progress=lambda value: print(json.dumps(value), flush=True))
    check_artifacts()
    result.update(panel_sha256=panel_sha256, checkpoint_sha256=checkpoint_sha256,
                  completed_training_steps=completed_steps,
                  parameters=sum(parameter.numel() for parameter in model.parameters()),
                  device=args.device, max_tokens=args.max_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "gates": result["gates"], "summary": result["summary"]}), flush=True)


if __name__ == "__main__":
    main()
