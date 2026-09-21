#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
question_output=${1:?output directory is required}
question_seconds=${2:-3600}
mkdir -p "$question_output"

.venv/bin/python - "$question_output" "$question_seconds" <<'PY'
import copy
import json
import sys
from pathlib import Path
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("The time-matched experiment requires CUDA.")
root = Path(sys.argv[1])
recipe = json.loads(Path("configs/question-learning.json").read_text())
for mode in ("fixed", "varied"):
    config = copy.deepcopy(recipe)
    config["defaults"]["question_mode"] = mode
    (root / f"{mode}-recipe.json").write_text(json.dumps(config, indent=2) + "\n")
(root / "environment.json").write_text(json.dumps({
    "torch": str(torch.__version__), "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
    "model": "LFM2.5-350M", "base": "models/lfm2.5-350m",
    "precision": "float32", "optimizer": "AdamW", "learning_rate": 1e-5,
    "training_seconds_per_condition": float(sys.argv[2]), "maximum_steps": 12000,
    "all_sources_per_update": False, "all_parameters_trainable": True,
    "fresh_updates_per_replay": 3, "replay_capacity_per_source_batches": 128,
    "comparison": "same initial weights, populations and primary/partner sampling; matched measured training wall time on the same GPU, not exact FLOPs",
}, indent=2) + "\n")
PY

for question_mode in fixed varied; do
  HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -u scripts/train_shared_prediction.py \
    --base models/lfm2.5-350m --recipe "$question_output/$question_mode-recipe.json" \
    --extension intrep.problems.shared_prediction.record_sources \
    --data-root . --output "$question_output/$question_mode" \
    --steps 12000 --training-seconds "$question_seconds" --device cuda --threads 4 \
    --optimizer adamw --learning-rate 0.00001 --max-grad-norm 1 \
    --checkpoint-interval 500 --evaluation-examples 128 --evaluation-interval 1000 \
    --native-controls --prompts configs/question-learning-prompts.json
done

.venv/bin/python - "$question_output" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
results = [json.loads((root / mode / "result.json").read_text()) for mode in ("fixed", "varied")]
assert results[0]["initial_parameters_sha256"] == results[1]["initial_parameters_sha256"]
assert (root / "fixed/evaluation-panel.json").read_bytes() == (root / "varied/evaluation-panel.json").read_bytes()
assert all(len(result["source_progress"]) == 12 for result in results)
assert all(result["parameters"] == result["trainable_parameters"] for result in results)
print(json.dumps({"stage": "comparison_complete", "steps": [result["completed_steps"] for result in results],
                  "training_seconds": [result["training_seconds"] for result in results]}), flush=True)
PY
