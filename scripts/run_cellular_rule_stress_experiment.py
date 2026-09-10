"""Run the preregistered bounded follow-up after clean-seed replication."""

import argparse
import hashlib
import json
from pathlib import Path

from intrep.problems.cellular_rule_inference.evaluate import evaluate
from intrep.problems.cellular_rule_inference.stress import evaluate_stress
from intrep.problems.cellular_rule_inference.training import (
    RuleInferenceTrainingConfig,
    load_checkpoint,
    train,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--original-validation", type=Path, required=True)
    parser.add_argument("--original-test", type=Path, required=True)
    parser.add_argument("--stress-validation", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args()
    probe = json.loads(args.stress_validation.read_text())
    if hashlib.sha256(args.clean_checkpoint[0].read_bytes()).hexdigest() != probe["checkpoint_sha256"]:
        raise ValueError("first clean checkpoint must match the frozen validation probe")
    baseline_model, baseline_payload = load_checkpoint(args.clean_checkpoint[0])
    if baseline_payload["config"]["model_seed"] != 31:
        raise ValueError("the declared validation trigger uses seed 31")
    noise = {row["noise_rate"]: row["methods"]["model"]["accuracy"]
             for row in probe["noise"] if row["context_count"] == 8}
    change = next(row["methods"] for row in probe["change"]
                  if row["noise_rate"] == 0.1 and row["post_change_count"] == 4)
    noise_loss = noise[0.0] - noise[0.1]
    stale_loss = change["model_current_only"]["accuracy"] - change["model"]["accuracy"]
    decisions = {"noise_accuracy_loss": noise_loss, "stale_history_accuracy_loss": stale_loss,
                 "train_noise": noise_loss >= 0.03, "train_change": stale_loss >= 0.03,
                 "trigger_threshold": 0.03, "training_steps": 6000,
                 "validation_checkpoint_sha256": probe["checkpoint_sha256"]}
    if (args.run_dir / "decisions.json").exists():
        raise FileExistsError("use a fresh follow-up run directory")
    write_json(args.run_dir / "decisions.json", decisions)
    print(json.dumps(decisions), flush=True)
    checkpoints = []
    for checkpoint in args.clean_checkpoint:
        _, payload = load_checkpoint(checkpoint)
        checkpoints.append((f"clean-seed{payload['config']['model_seed']}", checkpoint))
    for name, enabled, probability in (("noise-seed31", decisions["train_noise"], 0.0),
                                       ("change-seed31", decisions["train_change"], 0.5)):
        if enabled:
            config = RuleInferenceTrainingConfig(model=baseline_model.config, max_steps=6000,
                                                 observation_noise=True, rule_change_probability=probability)
            checkpoint = train(config, args.run_dir / name, device=args.device)
            checkpoints.append((name, checkpoint))
    # All training decisions and weights are fixed before looking at final tests.
    for name, checkpoint in checkpoints:
        print(json.dumps({"evaluating": name}), flush=True)
        stress = evaluate_stress(checkpoint, device=args.device,
                                 exclude_rules_from=(args.original_validation, args.original_test, args.stress_validation))
        write_json(args.run_dir / name / "stress-test.json", stress)
        if not name.startswith("clean-"):
            clean = evaluate(checkpoint, rule_seed=19101, data_seed=23001,
                             exclude_rules_from=args.original_validation, device=args.device)
            write_json(args.run_dir / name / "clean-test.json", clean)


if __name__ == "__main__":
    main()
