"""Verify and archive a rule-transfer checkpoint without retaining local weights."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.rule_transfer_data import file_digest
from intrep.problems.shared_prediction.rule_transfer_training import LESSON_KEY, RuleLessons, state_digest
from intrep.problems.shared_prediction.sources import build_sources
from intrep.problems.shared_prediction.training import load_checkpoint, parameter_digests


def rclone(arguments):
    completed = subprocess.run(["rclone", *arguments, "--s3-no-check-bucket"], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"rclone {arguments[0]} failed with exit code {completed.returncode}")
    return completed.stdout


def verify(directory, root):
    extensions = ["intrep.problems.shared_prediction.record_sources"]
    for extension in extensions:
        importlib.import_module(extension)
    result = json.loads((directory / "result.json").read_text())
    checkpoint = directory / "checkpoint.pt"
    digest = file_digest(checkpoint)
    if digest != result["checkpoint_sha256"]:
        raise ValueError("the checkpoint differs from the reported training result")
    model, tokenizer, payload = load_checkpoint(checkpoint, extensions=extensions)
    if state_digest(parameter_digests(model)) != result["final_parameters_sha256"]:
        raise ValueError("CPU-loaded model parameters differ from the training result")
    trainer = JointTrainer(model, payload["trainer"]["weights"], optimizer="adamw",
                           learning_rate=result["settings"]["learning_rate"])
    trainer.load_state_dict(payload["trainer"])
    sources = build_sources(model, tokenizer, payload["recipe"], root)
    for name, source in sources.items():
        source.load_state_dict(payload["sources"][name])
    metadata = payload["provenance"][LESSON_KEY]
    settings, lesson_info = metadata["settings"], metadata["lessons"]
    lessons = RuleLessons(sources["mnist"].reader, lesson_info["orders"], condition=settings["condition"],
                          seed=settings["seed"], batches=settings["batches"], manifest=lesson_info["manifest"])
    lessons.load_state_dict(payload["sources"][LESSON_KEY])
    restored = {**{name: source.state_dict() for name, source in sources.items()}, LESSON_KEY: lessons.state_dict()}
    if (state_digest(restored) != state_digest(payload["sources"])
            or trainer.steps != result["completed_steps"] or len(sources) != 12):
        raise ValueError("CPU restoration changed sampling state or training progress")
    return {"schema_version": "intrep.rule_transfer_cpu_verification.v1", "completed_steps": trainer.steps,
            "background_sources_restored": list(sources), "lesson_state_restored": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
            "parameter_digest_matches_training": True, "optimizer_state_entries": len(trainer.optimizer.state),
            "checkpoint": {"bytes": checkpoint.stat().st_size, "sha256": digest}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    parser.add_argument("--keep-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.prefix or any(part in ("", ".", "..") for part in args.prefix.split("/")):
        parser.error("use a nonempty relative archive prefix")
    if args.local_output.exists():
        raise FileExistsError("the collected result directory already exists")
    torch.set_num_threads(4)
    remote = f"r2:{os.environ['R2_BUCKET']}/{args.prefix}"
    if json.loads(rclone(["lsjson", remote, "--recursive", "--files-only"])):
        raise FileExistsError("the archive prefix is already in use")
    verified = verify(args.directory, args.data_root.resolve())
    (args.directory / "cpu-verification.json").write_text(json.dumps(verified, indent=2) + "\n")
    rclone(["copy", str(args.directory), remote, "--immutable", "--transfers", "2"])
    rclone(["check", str(args.directory), remote, "--download", "--one-way", "--checkers", "2"])
    shutil.copytree(args.directory, args.local_output, ignore=shutil.ignore_patterns("checkpoint.pt"))
    archive = {"schema_version": "intrep.rule_transfer_archive.v1", "prefix": args.prefix,
               "checkpoint": verified["checkpoint"], "verified": True,
               "byte_comparison": "rclone check --download --one-way", "local_checkpoint_retained": False,
               "restore": f"bash scripts/restore_r2_artifact.sh {args.prefix} DESTINATION"}
    archive_path = args.local_output / "archive.json"
    archive_path.write_text(json.dumps(archive, indent=2) + "\n")
    rclone(["copyto", str(archive_path), remote + "/archive.json", "--immutable"])
    if not args.keep_checkpoint:
        (args.directory / "checkpoint.pt").unlink()
    print(json.dumps({"stage": "checkpoint_archived", "prefix": args.prefix, "verified": True}), flush=True)


if __name__ == "__main__":
    main()
