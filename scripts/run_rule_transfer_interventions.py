"""Train matched A/B/control branches, audit them, and measure development transfer."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from intrep.problems.shared_prediction.rule_transfer_data import file_digest


def read_result(directory):
    return json.loads((directory / "result.json").read_text())


def script(name, arguments):
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / name), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common", type=Path, required=True, help="restored calibration directory, including checkpoint and reports")
    parser.add_argument("--panel-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-prefix", required=True)
    parser.add_argument("--milestones", nargs="+", type=int, default=[225, 450, 900, 1800])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if (not args.milestones or args.milestones != sorted(set(args.milestones)) or min(args.milestones) < 1
            or args.work.exists() and any(args.work.iterdir())):
        parser.error("use increasing positive milestones and an empty working directory")
    common = read_result(args.common)
    settings = common["settings"]
    if any(limit * settings["batches"][3] % 90 for limit in args.milestones):
        parser.error("milestones must complete whole 90-example tuition cycles")
    panel = args.panel_directory / "panel.json"
    if (not common["prerequisites_passed"] or common["condition"] != "calibration"
            or file_digest(args.common / "checkpoint.pt") != common["checkpoint_sha256"]
            or file_digest(panel) != settings["panel_sha256"]):
        parser.error("the restored common checkpoint and fixed panel must match a passing calibration")
    args.work.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": "intrep.rule_transfer_intervention_plan.v1", "milestones": args.milestones,
            "selection": "earliest common milestone at which both A and B pass development prerequisites",
            "control_updates": "same as the selected A/B milestone",
            "holdout_evaluation": "separate, after the procedure and any development-based follow-up are fixed",
            "settings_from_common": settings, "common_checkpoint_sha256": common["checkpoint_sha256"]}
    with (args.output / "plan.json").open("x") as handle:
        handle.write(json.dumps(plan, indent=2) + "\n")

    def train(name, limit):
        directory = args.work / name
        checkpoint = directory / "checkpoint.pt"
        initialize = ["--resume", checkpoint] if checkpoint.exists() else ["--common", args.common / "checkpoint.pt"]
        script("train_rule_transfer.py", [*initialize, "--condition", name,
            "--manifest", args.panel_directory / f"text-{name}.json",
            "--recipe", args.panel_directory / "development-recipe.json", "--panel", panel,
            "--output", directory, "--steps", limit, "--interval", limit,
            "--device", args.device, "--threads", args.threads,
            "--extension", "intrep.problems.shared_prediction.record_sources",
            "--seed", settings["seed"], "--learning-rate", settings["learning_rate"],
            "--batches", *settings["batches"], "--weights", *settings["weights"],
            "--prompts", "configs/question-learning-prompts.json"])
        return read_result(directory)

    selected = None
    comparisons = []
    for limit in args.milestones:
        results = {name: train(name, limit) for name in ("a", "b")}
        comparisons.append({"updates": limit, "prerequisites": {name: result["prerequisites"] for name, result in results.items()}})
        if all(result["prerequisites_passed"] for result in results.values()):
            selected = limit
            break
    conditions = ["a", "b"]
    if selected is not None:
        control = train("control", selected)
        conditions.append("control")
        script("audit_rule_transfer_training.py", ["--common", args.common, "--a", args.work / "a", "--b", args.work / "b",
                                                    "--control", args.work / "control", "--output", args.output / "training-audit.json"])
        if control["prerequisites_passed"]:
            for name in conditions:
                script("evaluate_rule_transfer.py", ["--checkpoint", args.work / name / "checkpoint.pt",
                    "--panel", panel, "--split", "development", "--order", "b" if name == "b" else "a",
                    "--extension", "intrep.problems.shared_prediction.record_sources",
                    "--output", args.output / f"{name}-development.json", "--device", args.device, "--threads", args.threads])
            script("compare_rule_transfer.py", ["--a", args.output / "a-development.json", "--b", args.output / "b-development.json",
                                                "--output", args.output / "development-comparison.json"])
    outcome = {"selected_updates": selected, "milestone_results": comparisons, "trained_conditions": conditions,
               "development_transfer_evaluated": (args.output / "development-comparison.json").exists(),
               "holdout_evaluated": False}
    (args.output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    for name in conditions:
        script("archive_rule_transfer.py", ["--directory", args.work / name, "--prefix", args.archive_prefix + "/" + name,
            "--local-output", args.output / name])
    print(json.dumps({"stage": "interventions_complete", **outcome}), flush=True)


if __name__ == "__main__":
    main()
