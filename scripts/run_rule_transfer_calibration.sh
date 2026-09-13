#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
transfer_output=${1:?result directory is required}
transfer_work=${2:?disposable working directory is required}
transfer_panel=${3:?prepared panel directory is required}
: "${R2_ENV_FILE:?R2_ENV_FILE is required}"
: "${INITIAL_ARCHIVE_PREFIX:?the verified initial checkpoint archive is required}"
: "${INITIAL_CHECKPOINT_SHA256:?the expected initial checkpoint hash is required}"
: "${TRANSFER_ARCHIVE_PREFIX:?a new result archive prefix is required}"
set -a
. "$R2_ENV_FILE"
set +a
export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Other
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"
mkdir -p "$transfer_output" "$transfer_work/initial"

rclone copy "r2:$R2_BUCKET/$INITIAL_ARCHIVE_PREFIX" "$transfer_work/initial" \
  --include /checkpoint.pt --include '/tokenizer/**' --s3-no-check-bucket --immutable

.venv/bin/python - "$transfer_work/initial/checkpoint.pt" "$transfer_output" <<'PY'
import json
import os
import sys
from pathlib import Path
import torch
import transformers
from intrep.problems.shared_prediction.rule_transfer_data import file_digest

if not torch.cuda.is_available():
    raise SystemExit("The calibration workload requires CUDA.")
if file_digest(Path(sys.argv[1])) != os.environ["INITIAL_CHECKPOINT_SHA256"]:
    raise SystemExit("The restored initial checkpoint failed its byte verification.")
environment = {"torch": str(torch.__version__), "transformers": transformers.__version__,
               "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
               "model": "LFM2.5-350M", "precision": "float32", "optimizer": "AdamW", "learning_rate": 1e-5,
               "maximum_updates": 12000, "maximum_training_seconds": 10800,
               "all_parameters_trainable": True, "background_sources_per_update": 12,
               "initial_checkpoint_sha256": os.environ["INITIAL_CHECKPOINT_SHA256"]}
(Path(sys.argv[2]) / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
PY

HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -u scripts/train_rule_transfer.py \
  --initialize "$transfer_work/initial/checkpoint.pt" --condition calibration \
  --recipe "$transfer_panel/development-recipe.json" --panel "$transfer_panel/panel.json" \
  --output "$transfer_work/calibration" --data-root . --device cuda --threads 4 \
  --extension intrep.problems.shared_prediction.record_sources \
  --steps 12000 --training-seconds 10800 --interval 500 --stop-when-calibrated \
  --prompts configs/question-learning-prompts.json

.venv/bin/python -u scripts/archive_rule_transfer.py \
  --directory "$transfer_work/calibration" --data-root . \
  --prefix "$TRANSFER_ARCHIVE_PREFIX/calibration" --local-output "$transfer_output/calibration"
