"""Compare no replay and 1:1 replay within an explicit worker time budget.

The budget includes calibration, both training invocations and comparison. Cloud
provisioning, transfer and checkpoint retrieval require a separate allocation
reserve. Evaluation covers complete populations unless a sample is explicit.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import transformers

from intrep.problems.shared_prediction.evaluation import paired_comparison
from intrep.problems.shared_prediction.full_evaluation import paired_full_comparison
from intrep.problems.shared_prediction.sources import source_configs
from intrep.problems.shared_prediction.training import TimeBudgetExceeded, train


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_condition(directory, expected_sources, replay_every):
    result = read_json(directory / "result.json")
    if (not all(result["evaluation_complete"].values()) or not result["comparison_complete"]
            or result["initial_step"] != 0 or result["completed_steps"] < 1
            or result["parameters"] != result["trainable_parameters"]):
        raise ValueError(f"{directory.name}: training and complete before/after evaluations are required")
    progress = result["experience_replay"]
    fresh, replay = progress["fresh_updates"], progress["replay_updates"]
    if (set(fresh) != set(expected_sources) or set(replay) != set(expected_sources)
            or any(count < 1 for count in fresh.values())
            or progress["fresh_updates_per_replay"] != replay_every
            or sum(fresh.values()) + sum(replay.values()) != result["completed_steps"]
            or (replay_every == 0 and sum(replay.values()) != 0)
            or (replay_every == 1 and sum(fresh.values()) - sum(replay.values()) not in (0, 1))):
        raise ValueError(f"{directory.name}: every source must participate with the declared replay schedule")
    if result["elapsed_seconds"] > result["time_budget_seconds"]:
        raise ValueError(f"{directory.name}: the invocation exceeded its elapsed-time budget")
    return result


def compare_conditions(output, *, check_budget):
    names = ("no_replay", "replay_1to1")
    sources = [row["name"] for row in source_configs(read_json(output / "recipe.json"))]
    results = {name: verify_condition(output / name, sources, int(name == "replay_1to1")) for name in names}
    if len({result["initial_parameters_sha256"] for result in results.values()}) != 1:
        raise ValueError("comparison requires identical initial parameters")
    for key in ("time_budget_seconds", "training_seconds_budget", "parameters", "device", "torch_version"):
        if results[names[0]][key] != results[names[1]][key]:
            raise ValueError(f"comparison conditions differ: {key}")
    for filename in ("recipe.json", "provenance.json", "evaluation-panel.json"):
        if read_json(output / names[0] / filename) != read_json(output / names[1] / filename):
            raise ValueError(f"comparison inputs differ: {filename}")
    initial = [read_json(output / name / "evaluation/step-000000.json") for name in names]
    if (initial[0]["sources"] != initial[1]["sources"]
            or initial[0]["generations"] != initial[1]["generations"]):
        raise ValueError("comparison requires identical initial evaluation results")
    final = [read_json(output / name / "evaluation" / f"step-{results[name]['completed_steps']:06d}.json")
             for name in names]
    if "rows_directory" in final[0]:
        paired = paired_full_comparison(
            final[0]["sources"], final[1]["sources"],
            before_directory=output / names[0] / "evaluation" / final[0]["rows_directory"],
            after_directory=output / names[1] / "evaluation" / final[1]["rows_directory"],
            check_budget=check_budget)
    else:
        paired = paired_comparison(final[0]["sources"], final[1]["sources"], check_budget=check_budget)
    return {"initial_parameters_sha256": results[names[0]]["initial_parameters_sha256"],
            "matched_recipe_provenance_evaluation_and_initial_results": True,
            "matched_time_budgets_and_device": True,
            "difference_direction": "replay_1to1 minus no_replay; compare each metric in its own units",
            "sources": paired,
            "conditions": {name: {key: results[name][key] for key in (
                "completed_steps", "training_seconds", "elapsed_seconds", "timing_seconds",
                "experience_replay", "evaluation_before", "evaluation_after", "paired_evaluation",
                "final_parameters_sha256", "stop_reason")} for name in names}}


def run_comparison(*, base, recipe, root, output, time_budget_seconds, device="cuda", threads=4,
                   learning_rate=1e-6, evaluation_examples=None, prompts=None, extensions=()):
    if isinstance(time_budget_seconds, bool) or not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0:
        raise ValueError("an explicit positive finite worker time budget is required")
    if threads < 1 or (evaluation_examples is not None and evaluation_examples < 1):
        raise ValueError("threads and an explicit evaluation sample must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is required for the requested device")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(threads)
    for module in extensions:
        importlib.import_module(module)
    names = [row["name"] for row in source_configs(recipe)]
    order = ["no_replay", "replay_1to1"]
    random.Random(recipe.get("seed", 47)).shuffle(order)
    plan = {"worker_time_budget_seconds": time_budget_seconds, "condition_order": order,
            "sources": names, "seed": recipe.get("seed", 47), "precision": "float32",
            "optimizer": "adamw", "learning_rate": learning_rate, "max_grad_norm": 1.0,
            "threads": threads, "replay_capacity_per_source_batches": 128,
            "evaluation_examples": evaluation_examples, "evaluation_scope": (
                "complete populations" if evaluation_examples is None else "explicit fixed development sample"),
            "calibration_updates": len(names), "calibration_maximum_fraction": 1 / 3,
            "checkpoint_schedule": "final checkpoint only, for calibration and each condition",
            "budget_scope": "worker calibration, training, evaluation, saves and comparison; cloud lifecycle is external"}
    write_json(output / "plan.json", plan)
    write_json(output / "recipe.json", recipe)
    write_json(output / "environment.json", {
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "device": device, "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
        "cuda": torch.version.cuda})
    summary = {"schema_version": "intrep.experience_replay_comparison.v1", "complete": False,
               "limitations": "One seed on development data; equal time caps are not equal FLOPs or long-term retention evidence."}

    def remaining():
        return time_budget_seconds - (time.perf_counter() - started)

    def check_budget():
        if remaining() <= 0:
            raise TimeBudgetExceeded("the comparison worker budget is exhausted")

    options = dict(base=base, recipe=recipe, root=root, device=device, optimizer="adamw",
                   learning_rate=learning_rate, max_grad_norm=1.0, checkpoint_interval=sys.maxsize,
                   evaluation_examples=evaluation_examples, prompts=prompts, extensions=extensions,
                   replay_capacity=128)

    def invoke(name, replay_every, **budgets):
        check_budget()
        train(**options, output=output / name, replay_every=replay_every, **budgets)
        result = verify_condition(output / name, names, replay_every)
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    try:
        # Touch every source once, including lazy optimizer state, before sizing the reserve.
        calibration = invoke("calibration", 0, steps=len(names), time_budget_seconds=remaining() / 3)
        calibration_seconds = time.perf_counter() - started
        overhead = calibration["elapsed_seconds"] - calibration["training_seconds"]
        longest_update = max(row["seconds"] for row in calibration["joint_updates"])
        # Allow another evaluation and update to absorb timing variation and finish the save.
        reserve = overhead + calibration["timing_seconds"]["evaluation"] / 2 + longest_update
        analysis_reserve = max(calibration["timing_seconds"]["other"], longest_update)
        per_condition = (remaining() - analysis_reserve) / 2
        update_budget = per_condition - reserve
        plan.update(calibration_elapsed_seconds=calibration_seconds,
                    measured_nontraining_seconds=overhead, per_condition_reserve_seconds=reserve,
                    comparison_reserve_seconds=analysis_reserve, condition_time_budget_seconds=per_condition,
                    condition_training_seconds=update_budget)
        write_json(output / "plan.json", plan)
        if update_budget <= calibration["training_seconds"] * 2:
            raise ValueError("the remaining budget cannot cover both conditions and complete final evaluations")
        for name in order:
            if remaining() < per_condition + analysis_reserve:
                raise TimeBudgetExceeded("not enough budget remains for an equal condition allocation")
            result = invoke(name, int(name == "replay_1to1"),
                            time_budget_seconds=per_condition, training_seconds=update_budget)
            if result["initial_parameters_sha256"] != calibration["initial_parameters_sha256"]:
                raise ValueError("calibration and comparison initial parameters differ")
            if result["stop_reason"] != "training_time_budget":
                raise ValueError("a condition could not finish its common update-time allocation")
        summary.update(compare_conditions(output, check_budget=check_budget))
        check_budget()
        summary["complete"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        summary.update(worker_elapsed_seconds=time.perf_counter() - started,
                       worker_time_budget_seconds=time_budget_seconds)
        write_json(output / "comparison.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-budget-seconds", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--evaluation-examples", type=int, help="explicit diagnostic sample; omission evaluates complete populations")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--extension", action="append", default=[])
    args = parser.parse_args()
    run_comparison(base=args.base, recipe=read_json(args.recipe), root=args.data_root.resolve(),
                   output=args.output, time_budget_seconds=args.time_budget_seconds, device=args.device,
                   threads=args.threads, learning_rate=args.learning_rate, evaluation_examples=args.evaluation_examples,
                   prompts=read_json(args.prompts) if args.prompts else None, extensions=args.extension)


if __name__ == "__main__":
    main()
