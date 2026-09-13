"""Run the fixed limited-image comparison, then evaluate every frozen holdout endpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from intrep.problems.shared_prediction.rule_transfer_data import file_digest
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_NAMES


def read_result(directory):
    return json.loads((directory / "result.json").read_text())


def read_trace(directory):
    path = directory / "steps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")


def script(name, arguments):
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / name), *map(str, arguments)], check=True)


def archive_endpoint(directory, prefix, output):
    # The completed endpoint remains available for holdout inference while a
    # CPU subprocess verifies and uploads a hard-linked snapshot.
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=directory.parent) as temporary:
        snapshot = Path(temporary) / directory.name
        shutil.copytree(directory, snapshot, copy_function=os.link)
        script("archive_rule_transfer.py", ["--directory", snapshot, "--prefix", prefix, "--local-output", output])


def extension_reason(result, previous):
    if result["prerequisites_passed"] or result["completed_steps"] != 1024:
        return []
    reasons = []
    if result["support_accuracy"] < .99:
        reasons.append("support_accuracy_below_99_percent")
    gain = result["prerequisites"]["new_image_rule"]["accuracy"] - previous["gates"]["new_image_rule"]["accuracy"]
    if gain >= .02 - 1e-12:
        reasons.append("development_gain_at_least_two_points_from_512_to_1024")
    return reasons


def audit_pair(parents, directories, manifest, manifest_sha256):
    results = {name: read_result(directory) for name, directory in directories.items()}
    traces = {name: read_trace(directory) for name, directory in directories.items()}
    count = len(manifest["examples"])
    lesson_names = (*LESSON_NAMES[:3], "image_tuition")
    comparable = lambda result: {key: value for key, value in result["settings"].items() if key != "condition"}
    reference = comparable(results["a"])
    for name, result in results.items():
        parent = read_result(parents / name)
        parent_trace = read_trace(parents / name)
        source_names = set(parent["source_progress"])
        if (result["condition"] != name or result["settings"]["condition"] != name
                or result["initial_checkpoint_sha256"] != parent["checkpoint_sha256"]
                or result["initial_parameters_sha256"] != parent["final_parameters_sha256"]
                or result["initial_background_state_sha256"] != parent_trace[-1]["background_state_sha256"]
                or result["starting_checkpoint_updates"] != parent["completed_steps"]
                or comparable(result) != reference or result["settings"]["image_manifest_sha256"] != manifest_sha256):
            raise ValueError("image branches must independently fork their recorded parents with matching settings")
        if (not result["optimizer_reset_at_fork"] or len(source_names) != 12
                or set(result["source_progress"]) != source_names
                or result["parameters"] != result["trainable_parameters"] or result["parameters"] != parent["parameters"]
                or result["lessons"]["image_manifest"] != manifest or result["lessons"]["manifest"] is not None):
            raise ValueError("image followup must preserve every parameter and source, replacing text tuition")
        updates = result["completed_steps"]
        presentations = updates * reference["batches"][3]
        if (result["image_teacher_budget"] != count or result["new_rule_image_training_presentations"] != presentations
                or result["new_rule_image_training_examples"] != min(presentations, count)
                or [row["step"] for row in traces[name]] != list(range(1, updates + 1))):
            raise ValueError("image exposure counters must match the complete update trace")
        for row in traces[name]:
            if (set(row["losses"]) != source_names | set(lesson_names) | {"weighted_loss", "grad_norm"}
                    or any(not math.isfinite(value) for value in row["losses"].values())
                    or set(row["lesson_inputs"]) != set(lesson_names)
                    or any(len(row["lesson_inputs"][lesson]) != reference["batches"][index]
                           for index, lesson in enumerate(lesson_names))):
                raise ValueError("each image update must include finite losses and inputs for all sources and lessons")
        seen = [identifier for row in traces[name] for identifier in row["lesson_inputs"]["image_tuition"]]
        expected = [manifest["examples"][index % count]["id"] for index in range(presentations)]
        if seen != expected:
            raise ValueError("image tuition input order or repetition counts differ")
        measurements = [json.loads(path.read_text()) for path in sorted((directories[name] / "prerequisites").glob("step-*.json"))]
        if (not measurements or any(row["split"] != "development" for row in measurements)
                or result["new_rule_image_evaluation_queries"] != sum(row["new_rule_image_queries"] for row in measurements)
                or result["prerequisites"] != measurements[-1]["gates"]):
            raise ValueError("selection must account for every development/support query and final gate")
        passing = [row["step"] for row in measurements if row["passed"]]
        if passing and updates != passing[0]:
            raise ValueError("select the first scheduled passing image checkpoint")
    for a, control in zip(traces["a"], traces["control"]):
        if a["background_state_sha256"] != control["background_state_sha256"] or a["lesson_inputs"] != control["lesson_inputs"]:
            raise ValueError("image branches must consume identical inputs over their shared update prefix")
    if results["a"]["initial_background_state_sha256"] != results["control"]["initial_background_state_sha256"]:
        raise ValueError("parent background readers must start at the same state")
    return {"schema_version": "intrep.rule_transfer_image_audit.v1", "verified": True, "image_teacher_budget": count,
            "image_manifest_sha256": manifest_sha256, "original_sources_per_update": 12, "all_parameters_trainable": True,
            "matched_shared_prefix_updates": min(len(trace) for trace in traces.values()),
            "matched_background_and_lesson_inputs": True, "new_text_tuition": False,
            "updates": {name: result["completed_steps"] for name, result in results.items()},
            "training_seconds": {name: result["training_seconds"] for name, result in results.items()},
            "image_presentations": {name: result["new_rule_image_training_presentations"] for name, result in results.items()},
            "image_example_exposures": {name: dict(Counter(identifier for row in trace for identifier in row["lesson_inputs"]["image_tuition"]))
                                       for name, trace in traces.items()},
            "prerequisites_passed": {name: result["prerequisites_passed"] for name, result in results.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--parent-audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--panel-directory", type=Path, required=True)
    parser.add_argument("--support-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-prefix", required=True)
    parser.add_argument("--isolate-timing", action="store_true",
                        help="defer CPU verification and uploads until all training and evaluation measurements finish")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    parent_audit = json.loads(args.parent_audit.read_text())
    panel = args.panel_directory / "panel.json"
    if (args.work.exists() and any(args.work.iterdir()) or not parent_audit["verified"]
            or not parent_audit["prerequisites_passed"] or plan["panel_sha256"] != file_digest(panel)
            or plan["teacher_budgets"] != [32, 128, 512] or plan["measurement_updates"] != [0, 64, 128, 256, 512, 1024]):
        parser.error("use an empty work directory, passing parent audit and the fixed image-followup protocol")
    parents = {name: read_result(args.parents / name) for name in ("a", "b", "control")}
    for name, result in parents.items():
        if (not result["prerequisites_passed"] or result["completed_steps"] != parent_audit["updates_per_condition"]
                or result["checkpoint_sha256"] != file_digest(args.parents / name / "checkpoint.pt")
                or result["new_rule_image_training_examples"] != 0):
            parser.error("all original endpoints must match their passing, text-only parent records")
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.plan, args.output / "protocol.json")
    with ThreadPoolExecutor(max_workers=1) as archiver:
        run_comparison(args, plan, parents, panel, archiver)


def run_comparison(args, plan, parents, panel, archiver):
    settings = parents["a"]["settings"]
    endpoints, extensions, archives = {}, {}, {}
    archive_schedule = "after-evaluation" if args.isolate_timing else "overlap-training-and-evaluation"
    for count in plan["teacher_budgets"]:
        manifest_path = args.support_directory / f"image-{count:04d}.json"
        manifest = json.loads(manifest_path.read_text())
        if len(manifest["examples"]) != count or manifest["panel_sha256"] != file_digest(panel):
            raise ValueError("the image support differs from the fixed budget or panel")
        directories = {}
        for name in ("a", "control"):
            for future in archives.values():
                if future.done():
                    future.result()
            key = f"image-{count:04d}-{name}"
            directory = args.work / key
            directories[name] = directory
            common_args = ["--condition", name, "--image-manifest", manifest_path,
                "--recipe", args.panel_directory / "development-recipe.json", "--panel", panel,
                "--output", directory, "--stop-when-adapted", "--device", args.device, "--threads", args.threads,
                "--extension", "intrep.problems.shared_prediction.record_sources",
                "--seed", settings["seed"], "--learning-rate", plan["optimizer"]["learning_rate"],
                "--batches", *[plan["training"]["batches"][lesson] for lesson in (*LESSON_NAMES[:3], "image_tuition")],
                "--weights", *[plan["training"]["weights"][lesson] for lesson in (*LESSON_NAMES[:3], "image_tuition")],
                "--prompts", "configs/question-learning-prompts.json"]
            script("train_rule_transfer.py", ["--common", args.parents / name / "checkpoint.pt", *common_args,
                "--steps", 1024, "--milestones", *plan["measurement_updates"][1:]])
            result = read_result(directory)
            previous = json.loads((directory / "prerequisites/step-000512.json").read_text()) if result["completed_steps"] == 1024 else None
            reasons = extension_reason(result, previous)
            extensions[key] = {"extend_to_2048": bool(reasons), "reasons": reasons}
            write_once(args.output / (key + "-extension.json"), extensions[key])
            if reasons:
                script("train_rule_transfer.py", ["--resume", directory / "checkpoint.pt", *common_args,
                    "--steps", 2048, "--milestones", 2048])
                result = read_result(directory)
            endpoints[key] = {"checkpoint_sha256": result["checkpoint_sha256"], "updates": result["completed_steps"],
                              "prerequisites_passed": result["prerequisites_passed"], "training_seconds": result["training_seconds"]}
            if not args.isolate_timing:
                archives[key] = archiver.submit(archive_endpoint, directory, args.archive_prefix + "/" + key, args.output / key)
        audited = audit_pair(args.parents, directories, manifest, file_digest(manifest_path))
        write_once(args.output / f"image-{count:04d}-audit.json", audited)
        print(json.dumps({"stage": "image_budget_complete", "budget": count, "audit": audited}), flush=True)
    selection = {"schema_version": "intrep.rule_transfer_final_selection.v1", "frozen_at": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": file_digest(args.plan), "panel_sha256": file_digest(panel),
                 "parents": {name: result["checkpoint_sha256"] for name, result in parents.items()},
                 "image_endpoints": endpoints, "extensions": extensions, "holdout_evaluated": False,
                 "archive_schedule": archive_schedule}
    write_once(args.output / "selection.json", selection)
    checkpoints = {**{name: args.parents / name / "checkpoint.pt" for name in parents},
                   **{name: args.work / name / "checkpoint.pt" for name in endpoints}}
    for name, checkpoint in checkpoints.items():
        for future in archives.values():
            if future.done():
                future.result()
        script("evaluate_rule_transfer.py", ["--checkpoint", checkpoint, "--panel", panel, "--split", "holdout",
            "--order", "b" if name == "b" else "a", "--extension", "intrep.problems.shared_prediction.record_sources",
            "--output", args.output / "holdout" / (name + ".json"), "--device", args.device, "--threads", args.threads])
    script("compare_rule_transfer.py", ["--a", args.output / "holdout/a.json", "--b", args.output / "holdout/b.json",
                                        "--output", args.output / "holdout/comparison.json"])
    if args.isolate_timing:
        for name in endpoints:
            archives[name] = archiver.submit(archive_endpoint, args.work / name,
                                            args.archive_prefix + "/" + name, args.output / name)
    for future in archives.values():
        future.result()
    for name in endpoints:
        (args.work / name / "checkpoint.pt").unlink()
    write_once(args.output / "outcome.json", {"endpoints": endpoints, "holdout_evaluated": True,
               "holdout_variants": list(checkpoints), "image_checkpoints_archived": True,
               "archive_schedule": archive_schedule})
    print(json.dumps({"stage": "image_followup_complete", "holdout_variants": list(checkpoints)}), flush=True)


if __name__ == "__main__":
    main()
