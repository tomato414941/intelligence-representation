"""Verify a shared checkpoint on CPU, archive it to R2, and collect small results."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch

from intrep.learning.joint import JointTrainer
from intrep.problems.shared_prediction.sources import build_sources
from intrep.problems.shared_prediction.streams import file_identity
from intrep.problems.shared_prediction.training import generate_text, load_checkpoint, parameter_digests


def verify_checkpoint(directory, data_root):
    result = json.loads((directory / "result.json").read_text())
    extensions = ["intrep.problems.shared_prediction.record_sources"]
    for extension in extensions:
        importlib.import_module(extension)
    model, tokenizer, state = load_checkpoint(directory / "checkpoint.pt", extensions=extensions)
    digest = hashlib.sha256(json.dumps(parameter_digests(model), sort_keys=True).encode()).hexdigest()
    if digest != result["final_parameters_sha256"]:
        raise ValueError("CPU-loaded parameters differ from the measured final model")
    trainer = JointTrainer(model, state["trainer"]["weights"], optimizer="adamw", learning_rate=1e-5)
    trainer.load_state_dict(state["trainer"])
    sources = build_sources(model, tokenizer, state["recipe"], data_root)
    for name, source in sources.items():
        source.load_state_dict(state["sources"][name])
    if trainer.steps != result["completed_steps"] or set(sources) != set(result["source_progress"]):
        raise ValueError("the checkpoint does not restore the final training progress")
    report = json.loads((directory / "generations" / f"step-{trainer.steps:06d}.json").read_text())
    selected = [row for row in report["generations"] if row["split"] == "development" and "expected" in row][:2]
    outputs = [{**row, "cpu_answer": generate_text(model, tokenizer, row["prompt"], max_tokens=row.get("max_tokens", 48))}
               for row in selected]
    verification = {
        "schema_version": "intrep.instruction-retention-cpu-verification.v1",
        "completed_steps": trainer.steps, "sources_restored": list(sources),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "final_parameters_sha256": digest, "parameter_digest_matches_training": True,
        "optimizer_state_entries": len(trainer.optimizer.state),
        "generations": outputs, "cpu_generation_matches_cuda": all(row["answer"] == row["cpu_answer"] for row in outputs),
        "checkpoint": file_identity(directory / "checkpoint.pt"),
    }
    (directory / "cpu-verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    return verification


def run_rclone(arguments):
    return subprocess.run(["rclone", *arguments, "--s3-no-check-bucket"], check=True,
                          capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--local-output", type=Path, required=True)
    args = parser.parse_args()
    if not args.prefix or any(part in {"", ".", ".."} for part in args.prefix.split("/")):
        parser.error("choose a nonempty relative archive prefix")
    if args.local_output.exists():
        raise FileExistsError("collection destination already exists")
    torch.set_num_threads(4)
    remote = f"r2:{os.environ['R2_BUCKET']}/{args.prefix}"
    if json.loads(run_rclone(["lsjson", remote, "--recursive", "--files-only"])):
        raise FileExistsError("the archive prefix already contains files")
    verified = verify_checkpoint(args.directory, args.data_root.resolve())
    run_rclone(["copy", str(args.directory), remote, "--immutable", "--transfers", "2"])
    # Compare the actual object contents, including multipart checkpoints.
    checked = subprocess.run(["rclone", "check", str(args.directory), remote, "--download", "--one-way",
                              "--checkers", "2", "--s3-no-check-bucket"], capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError(f"archive byte verification failed: {checked.stderr}")
    shutil.copytree(args.directory, args.local_output, ignore=shutil.ignore_patterns("checkpoint.pt"))
    archive = {"schema_version": "intrep.instruction-retention-archive.v1",
               "prefix": args.prefix, "checkpoint": verified["checkpoint"],
               "storage": "project R2 bucket; standard storage",
               "byte_comparison": "rclone check --download --one-way", "verified": True,
               "restore": f"bash scripts/restore_r2_artifact.sh {args.prefix} DESTINATION",
               "local_checkpoint_retained": False}
    (args.local_output / "archive.json").write_text(json.dumps(archive, indent=2) + "\n")
    (args.local_output / "archive-check.txt").write_text(checked.stdout + checked.stderr)
    run_rclone(["copyto", str(args.local_output / "archive.json"), remote + "/archive.json", "--immutable"])
    run_rclone(["copyto", str(args.local_output / "archive-check.txt"), remote + "/archive-check.txt", "--immutable"])
    # The disposable working checkpoint can be removed only after verified storage.
    (args.directory / "checkpoint.pt").unlink()
    print(json.dumps({"stage": "checkpoint_archived", "prefix": args.prefix,
                      "checkpoint_bytes": verified["checkpoint"]["bytes"], "verified": True}), flush=True)


if __name__ == "__main__":
    main()
