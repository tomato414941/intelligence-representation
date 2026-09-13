"""Calibrate or fork a fully trainable, twelve-source rule-transfer experiment."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import time
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.evaluation import evaluate_panel, make_panel
from intrep.problems.shared_prediction.recipe import evaluation_recipe, validate_recipe
from intrep.problems.shared_prediction.rule_transfer_data import file_digest, load_panel
from intrep.problems.shared_prediction.rule_transfer_training import (
    LESSON_KEY, LESSON_SLOTS, RuleLessons, measure_image_followup, measure_prerequisites, state_digest, validate_training_recipe,
)
from intrep.problems.shared_prediction.sources import Source, build_sources, source_configs
from intrep.problems.shared_prediction.training import (
    GradientAudit, generate_text, load_checkpoint, parameter_digests, save_checkpoint,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def train_trial(args):
    if (args.steps < 1 or args.interval < 1 or args.threads < 1 or min(args.batches) < 1
            or any(not math.isfinite(value) or value <= 0 for value in (*args.weights, args.learning_rate))
            or args.training_seconds is not None and not 0 < args.training_seconds < float("inf")):
        raise ValueError("training budgets, batches, weights and learning rate must be positive")
    if bool(args.initialize) != (args.condition == "calibration" and not args.resume):
        raise ValueError("initialize calibrates an existing shared checkpoint; common forks a calibrated checkpoint")
    if args.common and args.condition == "calibration":
        raise ValueError("a common checkpoint is for an intervention condition")
    if args.milestones and (args.milestones != sorted(set(args.milestones)) or min(args.milestones) < 1):
        raise ValueError("measurement milestones must be increasing positive update counts")
    if args.image_manifest and (args.condition not in ("a", "control") or args.manifest):
        raise ValueError("image followup uses A/control and replaces their text tuition")
    if args.stop_when_adapted and not args.image_manifest:
        raise ValueError("image-adaptation stopping requires an image manifest")
    if not args.image_manifest and (args.manifest is None) != (args.condition == "calibration"):
        raise ValueError("each intervention requires its prepared text manifest")
    source_path = args.initialize or args.common or args.resume
    checkpoint_path = args.output / "checkpoint.pt"
    if args.resume:
        if args.resume.resolve() != checkpoint_path.resolve():
            raise ValueError("resume continues in the original output directory")
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("a new trial requires an empty output directory")
    torch.set_num_threads(args.threads)
    for extension in args.extension:
        importlib.import_module(extension)
    root = args.data_root.resolve()
    recipe = json.loads(args.recipe.read_text())
    validate_recipe(recipe, root)
    panel, images, labels = load_panel(args.panel, root)
    panel_sha256 = file_digest(args.panel)
    source_sha256 = file_digest(source_path)
    settings = {"condition": args.condition, "learning_rate": args.learning_rate,
                "batches": args.batches, "weights": args.weights, "seed": args.seed,
                "panel_sha256": panel_sha256, "optimizer": "AdamW", "max_grad_norm": 1.0,
                "manifest_sha256": file_digest(args.manifest) if args.manifest else None}
    if args.image_manifest:
        settings["image_manifest_sha256"] = file_digest(args.image_manifest)
    model, tokenizer, payload = load_checkpoint(source_path, device=args.device, extensions=args.extension)
    validate_training_recipe(recipe, payload["recipe"], panel, root)
    if args.resume and recipe != payload["recipe"]:
        raise ValueError("exact continuation must preserve the entire recipe")
    sources = build_sources(model, tokenizer, recipe, root)
    provenance = {name: source.provenance() for name, source in sources.items()}
    for name, source in sources.items():
        if payload["provenance"][name] != provenance[name]:
            raise ValueError("background training data changed")
        source.load_state_dict(payload["sources"][name])
    lesson_source = sources["mnist"].reader
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    image_manifest = json.loads(args.image_manifest.read_text()) if args.image_manifest else None
    if image_manifest is not None:
        if image_manifest["panel_sha256"] != panel_sha256:
            raise ValueError("image tuition must use the same reserved development/holdout panel")
        for name in ("images", "labels"):
            entry = image_manifest["files"][name]
            path = lesson_source.path(name)
            if (root / entry["path"]).resolve() != path.resolve() or file_digest(path) != entry["sha256"]:
                raise ValueError("image tuition source files differ from the original training population")
    lessons = RuleLessons(lesson_source, panel["orders"], condition=args.condition, batches=args.batches,
                          seed=args.seed, manifest=manifest, image_manifest=image_manifest)
    # Check that all spoken candidates remain distinct single-token outputs.
    candidates = [*map(str, range(10)), "yes", "no"]
    candidate_ids = [lesson_source.text_ids(value) for value in candidates]
    if any(len(ids) != 1 for ids in candidate_ids) or len({tuple(ids) for ids in candidate_ids}) != len(candidates):
        raise ValueError("the rule experiment needs distinct single-token digit and yes/no answers")
    weights = {row["name"]: row.get("weight", 1.0) for row in source_configs(recipe)}
    weights.update({name: args.weights[LESSON_SLOTS[name]] for name in lessons.names})
    trainer = JointTrainer(model, weights, optimizer="adamw", learning_rate=args.learning_rate, max_grad_norm=1.)
    old_metadata = payload["provenance"].get(LESSON_KEY)
    if args.resume:
        if old_metadata is None or old_metadata["settings"] != settings:
            raise ValueError("exact continuation requires the original lesson settings")
        trainer.load_state_dict(payload["trainer"])
        lessons.load_state_dict(payload["sources"][LESSON_KEY])
        metadata = copy.deepcopy(old_metadata)
    else:
        if args.common:
            parent_condition = args.condition if image_manifest is not None else "calibration"
            if (old_metadata is None or old_metadata["settings"]["condition"] != parent_condition
                    or not old_metadata.get("prerequisites_passed")):
                raise ValueError("the parent checkpoint must pass its prerequisites and match this training phase")
            if "image_manifest_sha256" in old_metadata["settings"]:
                raise ValueError("each image budget must fork its original text-tuition/rehearsal parent")
            if (old_metadata["settings"]["panel_sha256"] != panel_sha256
                    or old_metadata["settings"]["batches"] != args.batches
                    or old_metadata["settings"]["seed"] != args.seed):
                raise ValueError("the common checkpoint must use the same panel and supplemental sampling")
            lessons.load_state_dict(payload["sources"][LESSON_KEY])
            if image_manifest is not None:
                lessons.tuition_position = 0
        metadata = {"schema_version": "intrep.rule_transfer_image_training.v1" if image_manifest else "intrep.rule_transfer_training.v1",
                    "settings": settings,
                    "initial_checkpoint_sha256": source_sha256,
                    "initial_background_state_sha256": state_digest({name: source.state_dict() for name, source in sources.items()}),
                    "initial_parameters_sha256": state_digest(parameter_digests(model)),
                    "starting_checkpoint_updates": payload["trainer"]["steps"],
                    "optimizer_reset_at_fork": True, "training_seconds": 0., "lessons": lessons.provenance()}
        if image_manifest is not None:
            metadata["new_rule_image_evaluation_queries"] = 0
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        if lesson_source.device.type != "cuda":
            raise ValueError("continuation of CUDA sampling requires CUDA")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    del payload
    if args.steps <= trainer.steps:
        raise ValueError("the requested budget must advance this condition")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "recipe.json", recipe)
    write_json(args.output / "settings.json", settings)
    evaluation = evaluation_recipe(recipe)
    evaluation["defaults"].update(question_evaluation_examples=8, question_evaluation_worlds=4)
    evaluation_sources = build_sources(model, tokenizer, evaluation, root)
    background_panel = make_panel(evaluation_sources, 32)
    write_json(args.output / "background-panel.json", background_panel)
    prompts = json.loads(args.prompts.read_text()) if args.prompts else []
    readout_source = Source(model, tokenizer, {}, root)

    def background_measure():
        reports = evaluate_panel(model, evaluation_sources, background_panel, generate_answers=True)
        generations = [{**case, "answer": generate_text(model, tokenizer, case["prompt"], max_tokens=case.get("max_tokens", 48))}
                       for case in prompts]
        write_json(args.output / "background" / f"step-{trainer.steps:06d}.json",
                   {"step": trainer.steps, "sources": reports, "generations": generations})

    def measure():
        progress = lambda value: print(json.dumps(value), flush=True)
        if image_manifest is not None:
            result = measure_image_followup(readout_source, panel, images, labels, lessons,
                                            condition=args.condition, progress=progress)
            metadata["new_rule_image_evaluation_queries"] += result["new_rule_image_queries"]
        else:
            result = measure_prerequisites(readout_source, panel, images, labels, condition=args.condition, progress=progress)
        result.update(step=trainer.steps, panel_sha256=panel_sha256)
        write_json(args.output / "prerequisites" / f"step-{trainer.steps:06d}.json", result)
        metadata["prerequisites_passed"] = result["passed"]
        metadata["prerequisites_step"] = trainer.steps
        print(json.dumps({"stage": "image_followup_prerequisites" if image_manifest else "prerequisites",
                          "step": trainer.steps, "gates": result["gates"],
                          **({"support_accuracy": result["support_accuracy"]} if image_manifest else {})}), flush=True)
        return result

    def save():
        provenance[LESSON_KEY] = metadata
        save_checkpoint(checkpoint_path, model, trainer, {**sources, LESSON_KEY: lessons}, recipe,
                        provenance, tokenizer, args.extension)

    if not args.resume:
        background_measure()
        measured = measure()
    else:
        measured = json.loads((args.output / "prerequisites" / f"step-{trainer.steps:06d}.json").read_text())
    before = parameter_digests(model)
    audit = GradientAudit(model)
    audit_records = None
    invocation_seconds = 0.
    stopping = "step_budget"
    try:
        while trainer.steps < args.steps:
            if args.stop_when_calibrated and args.condition == "calibration" and measured["passed"]:
                stopping = "calibration_passed"
                break
            if args.stop_when_adapted and measured["passed"]:
                stopping = "image_prerequisites_passed"
                break
            if args.training_seconds is not None and invocation_seconds >= args.training_seconds:
                stopping = "training_time_budget"
                break
            if lesson_source.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            def background_loss(name, source):
                if audit:
                    audit.source = name
                return source.loss()

            def lesson_loss(name):
                if audit:
                    audit.source = name
                return lessons.loss(name)

            callbacks = {name: (lambda name=name, source=source: background_loss(name, source)) for name, source in sources.items()}
            callbacks.update({name: (lambda name=name: lesson_loss(name)) for name in lessons.names})
            metrics = trainer.step(callbacks)
            if audit:
                audit_records = audit.records
                audit.close()
                audit = None
            if lesson_source.device.type == "cuda":
                torch.cuda.synchronize()
            trace = state_digest({name: source.state_dict() for name, source in sources.items()})
            seconds = time.perf_counter() - start
            invocation_seconds += seconds
            metadata["training_seconds"] += seconds
            record = {"step": trainer.steps, "seconds": seconds, "losses": metrics,
                      "background_state_sha256": trace, "lesson_inputs": lessons.last_trace}
            with (args.output / "steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if trainer.steps <= 2 or trainer.steps % 25 == 0:
                print(json.dumps({"stage": "training", **{key: value for key, value in record.items() if key != "lesson_inputs"}}), flush=True)
            measurement_due = trainer.steps in args.milestones if args.milestones else trainer.steps % args.interval == 0
            if measurement_due:
                measured = measure()
                save()
    finally:
        if audit:
            audit.close()
    if measured["step"] != trainer.steps:
        measured = measure()
    if args.stop_when_calibrated and args.condition == "calibration" and measured["passed"]:
        stopping = "calibration_passed"
    if args.stop_when_adapted and measured["passed"]:
        stopping = "image_prerequisites_passed"
    after = parameter_digests(model)
    metadata["final_parameters_sha256"] = state_digest(after)
    save()
    background_measure()
    result = {**metadata, "condition": args.condition, "completed_steps": trainer.steps,
              "requested_steps": args.steps, "stopping_reason": stopping,
              "invocation_training_seconds": invocation_seconds, "prerequisites": measured["gates"],
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
              "changed_core_parameter_tensors": sum(name.startswith("core.") and before[name] != after[name] for name in before),
              "source_progress": {name: source.progress() for name, source in sources.items()},
              "gradient_audit_first_update": audit_records,
              "checkpoint_sha256": file_digest(checkpoint_path),
              "new_rule_image_training_examples": min(lessons.tuition_position, len(image_manifest["examples"])) if image_manifest else 0,
              "new_rule_image_evaluation_queries": metadata.get("new_rule_image_evaluation_queries", 0)}
    if image_manifest is not None:
        result.update(new_rule_image_training_presentations=lessons.tuition_position,
                      image_teacher_budget=len(image_manifest["examples"]), support_accuracy=measured["support_accuracy"])
    write_json(args.output / "result.json", result)
    print(json.dumps({"stage": "trial_complete", "condition": args.condition, "step": trainer.steps,
                      "stopping_reason": stopping, "prerequisites_passed": measured["passed"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    initialize = parser.add_mutually_exclusive_group(required=True)
    initialize.add_argument("--initialize", type=Path)
    initialize.add_argument("--common", type=Path)
    initialize.add_argument("--resume", type=Path)
    parser.add_argument("--condition", choices=("calibration", "a", "b", "control"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image-manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--milestones", nargs="+", type=int)
    parser.add_argument("--stop-when-calibrated", action="store_true")
    parser.add_argument("--stop-when-adapted", action="store_true")
    parser.add_argument("--batches", nargs=4, type=int, default=[16, 8, 8, 8])
    parser.add_argument("--weights", nargs=4, type=float, default=[8., 8., 2., 8.])
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--extension", action="append", default=[])
    parser.add_argument("--prompts", type=Path)
    train_trial(parser.parse_args())


if __name__ == "__main__":
    main()
