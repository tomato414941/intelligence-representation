#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
experiment_output=${1:?output directory is required}
experiment_steps=${2:-1000}
mkdir -p "$experiment_output"

.venv/bin/python - "$experiment_output/environment.json" <<'PY'
import json
import sys
from pathlib import Path
import torch
import transformers

if not torch.cuda.is_available():
    raise SystemExit("This experiment requires its measured CUDA environment.")
torch.cuda.reset_peak_memory_stats()
Path(sys.argv[1]).write_text(json.dumps({
    "torch": str(torch.__version__), "transformers": transformers.__version__,
    "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
    "precision": "float32", "optimizer": "AdamW", "learning_rate": 1e-5,
    "all_sources_per_update": False, "all_parameters_trainable": True,
    "fresh_updates_per_replay": 3, "replay_capacity_per_source_batches": 128,
}, indent=2) + "\n")
PY

for experiment_size in 230m 350m; do
  HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -u scripts/train_shared_prediction.py \
    --base "models/lfm2.5-$experiment_size" --recipe configs/joint-lfm.json \
    --data-root . --output "$experiment_output/$experiment_size" \
    --steps "$experiment_steps" --device cuda --threads 4 \
    --optimizer adamw --learning-rate 0.00001 --max-grad-norm 1 \
    --checkpoint-interval 250 --evaluation-examples 128 --evaluation-interval 250 \
    --native-controls --prompts configs/joint-lfm-evaluation-prompts.json
done
