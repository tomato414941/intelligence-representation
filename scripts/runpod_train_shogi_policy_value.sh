#!/usr/bin/env bash
set -euo pipefail

# Use RunPod as disposable compute.
# Project-specific RunPod operating notes live in docs/runpod.md.
# Keep this on container disk for the current KISS flow; network-volume
# reevaluation is tracked in issues/runpod-network-volume-revisit.md.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
DATA_SELECTION=${DATA_SELECTION:-data/shogi/training-data-bundles/qhapaq-full/data-selection.json}
TENSOR_CACHE=${TENSOR_CACHE:-}
INIT_CHECKPOINT_PATH=${INIT_CHECKPOINT_PATH:-}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-shogi-policy-value}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
SECURE_CLOUD=${SECURE_CLOUD:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-1.0}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-65536}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-}
LOG_EVERY=${LOG_EVERY:-100}
# Save progress before final sync so interrupted disposable pods do not lose the
# whole training run.
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
METRICS_EVERY=${METRICS_EVERY:-1000}
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-3}
EVAL_EVERY=${EVAL_EVERY:-1000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-}
EMBEDDING_DIM=${EMBEDDING_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-6}
MODEL=${MODEL:-shared_transformer}
# Optional RunPod data-center pin. See docs/runpod.md before long baselines.
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

.venv/bin/python - "$MODEL" <<'PY'
import sys

from intrep.problems.shogi_policy_value.model import SHOGI_POLICY_VALUE_MODEL_NAMES

model = sys.argv[1]
if model not in SHOGI_POLICY_VALUE_MODEL_NAMES:
    names = ", ".join(SHOGI_POLICY_VALUE_MODEL_NAMES)
    raise SystemExit(f"unsupported MODEL={model!r}; expected one of: {names}")
PY

TRAINING_INPUT_ARGS=(--data-selection "$DATA_SELECTION")
if [[ -n "$TENSOR_CACHE" ]]; then
  TRAINING_INPUT_ARGS+=(--tensor-cache "$TENSOR_CACHE")
fi
mapfile -t TRAINING_INPUT_FILES < <(
  .venv/bin/python -m intrep.problems.shogi_policy_value.training_inputs "${TRAINING_INPUT_ARGS[@]}"
)
SYNC_ARGS=()
for input_file in "${TRAINING_INPUT_FILES[@]}"; do
  SYNC_ARGS+=(--sync "$input_file")
done
if [[ -n "$INIT_CHECKPOINT_PATH" ]]; then
  SYNC_ARGS+=(--sync "$INIT_CHECKPOINT_PATH")
fi
CLOUD_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  CLOUD_ARGS+=(--secure-cloud)
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$PWD" \
  --name intrep-shogi-policy-value \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${CLOUD_ARGS[@]}" \
  ${DATA_CENTER_IDS:+--data-center-ids "$DATA_CENTER_IDS"} \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 600 \
  --ssh-wait-seconds 180 \
  --allow-existing-pods \
  --sync src \
  --sync tests \
  --sync pyproject.toml \
  --sync uv.lock \
  --sync README.md \
  --sync AGENTS.md \
  --sync scripts/setup_runpod.sh \
  "${SYNC_ARGS[@]}" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"
echo \"run_config max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT embedding_dim=$EMBEDDING_DIM hidden_dim=$HIDDEN_DIM num_heads=$NUM_HEADS num_layers=$NUM_LAYERS model=$MODEL num_workers=$NUM_WORKERS max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES checkpoint_every=$CHECKPOINT_EVERY metrics_every=$METRICS_EVERY keep_last_n_checkpoints=$KEEP_LAST_N_CHECKPOINTS eval_every=$EVAL_EVERY early_stopping_patience=$EARLY_STOPPING_PATIENCE tensor_cache=$TENSOR_CACHE init_checkpoint_path=$INIT_CHECKPOINT_PATH\"
TRAIN_ARGS=()
if [[ -n \"$TENSOR_CACHE\" ]]; then
  TRAIN_ARGS+=(--tensor-cache \"$TENSOR_CACHE\")
fi
if [[ -n \"$INIT_CHECKPOINT_PATH\" ]]; then
  TRAIN_ARGS+=(--init-checkpoint-path \"$INIT_CHECKPOINT_PATH\")
fi
if [[ -n \"$EARLY_STOPPING_PATIENCE\" ]]; then
  TRAIN_ARGS+=(--early-stopping-patience \"$EARLY_STOPPING_PATIENCE\")
fi
if [[ -n \"$MAX_TRAIN_EVAL_EXAMPLES\" ]]; then
  TRAIN_ARGS+=(--max-train-eval-examples \"$MAX_TRAIN_EVAL_EXAMPLES\")
fi
if [[ -n \"$MAX_EVAL_EXAMPLES\" ]]; then
  TRAIN_ARGS+=(--max-eval-examples \"$MAX_EVAL_EXAMPLES\")
fi
.venv/bin/python -u -m intrep.train_shogi_policy_value \
  --data-selection \"$DATA_SELECTION\" \
  --checkpoint-path \"$OUTPUT_DIR/checkpoint.pt\" \
  --best-checkpoint-path \"$OUTPUT_DIR/best_checkpoint.pt\" \
  --metrics-path \"$OUTPUT_DIR/metrics.json\" \
  --max-steps \"$MAX_STEPS\" \
  --batch-size \"$BATCH_SIZE\" \
  --learning-rate \"$LEARNING_RATE\" \
  --weight-decay 0.01 \
  --embedding-dim \"$EMBEDDING_DIM\" \
  --hidden-dim \"$HIDDEN_DIM\" \
  --num-heads \"$NUM_HEADS\" \
  --num-layers \"$NUM_LAYERS\" \
  --model \"$MODEL\" \
  --policy-loss-weight \"$POLICY_LOSS_WEIGHT\" \
  --value-loss-weight \"$VALUE_LOSS_WEIGHT\" \
  --device cuda \
  --log-every \"$LOG_EVERY\" \
  --eval-every \"$EVAL_EVERY\" \
  --num-workers \"$NUM_WORKERS\" \
  --pin-memory \
  --checkpoint-every \"$CHECKPOINT_EVERY\" \
  --metrics-every \"$METRICS_EVERY\" \
  --keep-last-n-checkpoints \"$KEEP_LAST_N_CHECKPOINTS\" \
  \"\${TRAIN_ARGS[@]}\"" \
  "$@"
