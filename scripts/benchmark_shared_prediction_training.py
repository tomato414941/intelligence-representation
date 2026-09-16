"""Measure complete rule-transfer updates from a fixed checkpoint and sampling state."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import statistics
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.rule_transfer_data import file_digest
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_KEY, RuleLessons, state_digest
from intrep.problems.shared_prediction.sources import build_sources, source_configs
from intrep.problems.shared_prediction.training import load_checkpoint, parameter_digests


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def scale_recipe(recipe, multiplier):
    if type(multiplier) is not int or multiplier < 1:
        raise ValueError("batch multiplier must be a positive integer")
    result = copy.deepcopy(recipe)
    for row, config in zip(result["sources"], source_configs(recipe)):
        row["records_per_update"] = config["records_per_update"] * multiplier
    return result


def tensor_reference(model, path, *, create):
    """Check every parameter and clipped gradient after one identical update."""
    tensors = {"parameters": dict(model.named_parameters()),
               "gradients": {name: parameter.grad for name, parameter in model.named_parameters()}}
    if create:
        if path.exists():
            raise FileExistsError("the first-update tensor reference already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({kind: {name: value.detach().cpu() if value is not None else None
                          for name, value in values.items()} for kind, values in tensors.items()}, path)
        return {"created": True}
    reference = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    summary = {}
    for kind, values in tensors.items():
        if set(reference[kind]) != set(values):
            raise ValueError("the tensor reference has different parameter names")
        error_squared, reference_squared, maximum_error, count = 0., 0., 0., 0
        for name, value in values.items():
            expected = reference[kind][name]
            if (value is None) != (expected is None):
                raise ValueError("the tensor reference has different active gradients")
            if value is None:
                continue
            actual = value.detach().cpu()
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise ValueError("the tensor reference has different shapes or nonfinite values")
            difference = actual.double() - expected.double()
            error_squared += float(difference.square().sum())
            reference_squared += float(expected.double().square().sum())
            maximum_error = max(maximum_error, float(difference.abs().max()))
            count += actual.numel()
        relative_error = math.sqrt(error_squared / reference_squared) if reference_squared else math.sqrt(error_squared)
        limit = 1e-6 if kind == "parameters" else 1e-4
        summary[kind] = {"elements": count, "relative_l2_error": relative_error,
                         "maximum_absolute_error": maximum_error, "relative_l2_limit": limit}
        if relative_error > limit:
            raise ValueError(f"first-update {kind} differ beyond the declared tolerance: {relative_error}")
    return summary


def benchmark(args):
    started = time.perf_counter()
    if args.steps < 1 or args.warmup < 1 or args.threads < 1:
        raise ValueError("measurement, warmup and thread counts must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("benchmark output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    extensions = ["intrep.problems.shared_prediction.record_sources"]
    for extension in extensions:
        importlib.import_module(extension)
    if file_digest(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("benchmark checkpoint bytes differ from the selected archive")
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=args.device, extensions=extensions)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("the benchmark requires full-parameter training")
    recipe = scale_recipe(payload["recipe"], args.batch_multiplier)
    sources = build_sources(model, tokenizer, recipe, args.data_root.resolve())
    if len(sources) != 12:
        raise ValueError("the benchmark must retain all twelve original data sources")
    provenance = {name: source.provenance() for name, source in sources.items()}
    for name, source in sources.items():
        if provenance[name] != payload["provenance"][name]:
            raise ValueError(f"training population changed for {name}")
        source.load_state_dict(payload["sources"][name])
    metadata = payload["provenance"][LESSON_KEY]
    settings, lesson_info = metadata["settings"], metadata["lessons"]
    lesson_batches = [count * args.batch_multiplier for count in settings["batches"]]
    lessons = RuleLessons(sources["mnist"].reader, lesson_info["orders"], condition=settings["condition"],
                          seed=settings["seed"], batches=lesson_batches, manifest=lesson_info["manifest"],
                          image_manifest=lesson_info.get("image_manifest"))
    lessons.load_state_dict(payload["sources"][LESSON_KEY])
    trainer = JointTrainer(model, payload["trainer"]["weights"], optimizer="adamw",
                           learning_rate=settings["learning_rate"], max_grad_norm=payload["trainer"]["max_grad_norm"])
    trainer.load_state_dict(payload["trainer"])
    initial = {"parameters_sha256": state_digest(parameter_digests(model)),
               "sources_sha256": state_digest({name: source.state_dict() for name, source in sources.items()}),
               "lessons_sha256": state_digest(lessons.state_dict()),
               "optimizer_sha256": state_digest(payload["trainer"]), "checkpoint_steps": trainer.steps}
    torch.set_rng_state(payload["torch_rng"])
    if args.device.startswith("cuda"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    del payload
    callbacks = {name: source.loss for name, source in sources.items()}
    callbacks.update({name: lambda name=name: lessons.loss(name) for name in lessons.names})
    body_counts = {"calls": 0, "positions": 0}

    def body_call(_module, inputs):
        body_counts["calls"] += 1
        body_counts["positions"] += inputs[0].shape[0] * inputs[0].shape[1]

    handle = model.core.register_forward_pre_hook(body_call)
    rows, parity = [], None
    setup_seconds = time.perf_counter() - started
    try:
        with (args.output / "steps.jsonl").open("w") as trace, ExitStack() as profiling:
            for index in range(args.warmup + args.steps):
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                    if index == args.warmup:
                        torch.cuda.reset_peak_memory_stats()
                profile_active = getattr(args, "profile", False) and index >= args.warmup
                if profile_active and index == args.warmup:
                    from scripts.shared_prediction_profile import TrainingProfile
                    profiling.enter_context(TrainingProfile(trainer, callbacks, args.output, device=args.device))
                body_counts.update(calls=0, positions=0)
                start = time.perf_counter()
                with torch.profiler.record_function("intrep/update") if profile_active else nullcontext():
                    metrics = trainer.step(callbacks)
                    with torch.profiler.record_function("intrep/final_sync") if profile_active else nullcontext():
                        if args.device.startswith("cuda"):
                            torch.cuda.synchronize()
                    with torch.profiler.record_function("intrep/source_hash") if profile_active else nullcontext():
                        digest = state_digest({name: source.state_dict() for name, source in sources.items()})
                elapsed = time.perf_counter() - start
                row = {"index": index, "measured": index >= args.warmup, "seconds": elapsed,
                       "losses": metrics, "source_state_sha256": digest, "body": body_counts.copy(),
                       "sources": {name: source.last_update_info.copy() for name, source in sources.items()},
                       "lesson_inputs": copy.deepcopy(lessons.last_trace)}
                rows.append(row)
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                trace.flush()
                if index == 0 and args.tensor_reference:
                    parity = tensor_reference(model, args.tensor_reference, create=args.write_tensor_reference)
                if index == 0 or (index + 1) % 8 == 0:
                    print(json.dumps({"stage": "benchmark_update", "index": index,
                                      "batch_multiplier": args.batch_multiplier, "seconds": elapsed}), flush=True)
    finally:
        handle.remove()
    measured = [row for row in rows if row["measured"]]
    seconds = sum(row["seconds"] for row in measured)
    package = Path(importlib.import_module("intrep").__file__).parent
    environment = {"torch": str(torch.__version__), "transformers": importlib.import_module("transformers").__version__,
                   "device": args.device, "threads": args.threads, "dtype": str(next(model.parameters()).dtype),
                   "parameters": sum(parameter.numel() for parameter in model.parameters()),
                   "all_parameters_trainable": True, "cuda": torch.version.cuda,
                   "attention_implementation": model.core.body.config._attn_implementation,
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                   "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                   "source_files_sha256": {str(path.relative_to(package)): file_digest(path) for path in sorted(package.rglob("*.py"))}}
    if args.device.startswith("cuda"):
        environment.update(gpu=torch.cuda.get_device_name(0), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    result = {"schema_version": "intrep.training_benchmark.v1", "checkpoint_sha256": args.checkpoint_sha256,
              "revision": args.revision, "initial": initial, "environment": environment,
              "batch_multiplier": args.batch_multiplier, "warmup_updates": args.warmup,
              "profiled": bool(getattr(args, "profile", False)),
              "measured_updates": args.steps, "setup_seconds": setup_seconds,
              "training_seconds": seconds, "median_update_seconds": statistics.median(row["seconds"] for row in measured),
              "equivalent_base_updates_per_second": args.steps * args.batch_multiplier / seconds,
              "sequence_positions_per_second": sum(row["body"]["positions"] for row in measured) / seconds,
              "records_per_source": {name: sum(row["sources"][name]["records"] for row in measured) for name in sources},
              "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else None,
              "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if args.device.startswith("cuda") else None,
              "first_update_tensor_comparison": parity, "lesson_batches": lesson_batches,
              "total_seconds": time.perf_counter() - started,
              "scope": "Complete updates including input reads, all source/lesson losses, backward, AdamW and source-state hashing; excludes evaluation and checkpoint I/O. Batch sweeps measure throughput, not model quality."}
    write_json(args.output / "recipe.json", recipe)
    write_json(args.output / "result.json", result)
    print(json.dumps({"stage": "benchmark_complete", "output": str(args.output),
                      "training_seconds": seconds, "peak_mib": result["cuda_peak_allocated_mib"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-multiplier", type=int, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tensor-reference", type=Path)
    parser.add_argument("--write-tensor-reference", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Record CPU/CUDA traces; these timings include profiler overhead")
    args = parser.parse_args()
    if args.write_tensor_reference and not args.tensor_reference:
        parser.error("writing the tensor reference requires its path")
    try:
        benchmark(args)
    except torch.cuda.OutOfMemoryError:
        write_json(args.output / "failure.json", {"reason": "cuda_out_of_memory", "batch_multiplier": args.batch_multiplier})
        raise SystemExit(42)


if __name__ == "__main__":
    main()
