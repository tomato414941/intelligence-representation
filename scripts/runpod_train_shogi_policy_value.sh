#!/usr/bin/env bash
set -euo pipefail

# Train a shogi policy/value model on a RunPod GPU Pod.
# Tensor cache can be restored from R2 before training.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
RUNPOD_JOB_NAME=${RUNPOD_JOB_NAME:-intrep-shogi-policy-value-train}
GPU_TYPE=${GPU_TYPE:-NVIDIA RTX 4000 Ada Generation}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-720}
WAIT_SECONDS=${WAIT_SECONDS:-600}
SSH_WAIT_SECONDS=${SSH_WAIT_SECONDS:-180}
REMOTE_POLL_SECONDS=${REMOTE_POLL_SECONDS:-30}
SECURE_CLOUD=${SECURE_CLOUD:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
KEEP_POD=${KEEP_POD:-0}

ASSEMBLY_SPEC=${ASSEMBLY_SPEC:-}
LOCAL_BUNDLE=${LOCAL_BUNDLE:-data/shogi/training-data-bundles/qhapaq-full}
DATA_SELECTION=${DATA_SELECTION:-"$LOCAL_BUNDLE/data-selection.json"}
TENSOR_CACHE=${TENSOR_CACHE:-"$LOCAL_BUNDLE/cache/$ASSEMBLY_SPEC"}
R2_CACHE_PREFIX=${R2_CACHE_PREFIX:-}
R2_ENV_FILE=${R2_ENV_FILE:-"$HOME/.secrets/intrep-cloudflare-r2"}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-policy-value-train-$RUN_ID}

MAX_STEPS=${MAX_STEPS:-100000}
BATCH_SIZE=${BATCH_SIZE:-512}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
EMBEDDING_DIM=${EMBEDDING_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-6}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-1.0}
NUM_WORKERS=${NUM_WORKERS:-2}
LOG_EVERY=${LOG_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-1000}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-10}
DISABLE_EARLY_STOPPING=${DISABLE_EARLY_STOPPING:-0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
METRICS_EVERY=${METRICS_EVERY:-1000}
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-3}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-}

if [[ -z "$ASSEMBLY_SPEC" ]]; then
  echo "ASSEMBLY_SPEC is required for training" >&2
  exit 1
fi
if [[ -z "$R2_CACHE_PREFIX" ]]; then
  R2_CACHE_PREFIX="shogi/tensor-caches/$(basename "$LOCAL_BUNDLE")/$ASSEMBLY_SPEC"
fi
if [[ ! -f "$R2_ENV_FILE" ]]; then
  echo "R2_ENV_FILE not found: $R2_ENV_FILE" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo ".venv/bin/python is required for local input discovery" >&2
  exit 1
fi

.venv/bin/python - "$ASSEMBLY_SPEC" <<'PY'
import sys

from intrep.representation.assembly_specs.shogi_policy_value import SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS

assembly_spec = sys.argv[1]
if assembly_spec not in SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS:
    names = ", ".join(SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS)
    raise SystemExit(f"unsupported ASSEMBLY_SPEC={assembly_spec!r}; expected one of: {names}")
PY

SYNC_ARGS=(
  --sync src
  --sync pyproject.toml
  --sync uv.lock
  --sync README.md
  --sync AGENTS.md
  --sync scripts/setup_runpod.sh
  --sync scripts/restore_r2_artifact.sh
  --sync "$DATA_SELECTION"
)

R2_REMOTE_ENV="runs/shogi/.runpod-r2-env-$RUN_ID"
cleanup_r2_remote_env() {
  rm -f "$R2_REMOTE_ENV"
}
trap cleanup_r2_remote_env EXIT
mkdir -p "$(dirname "$R2_REMOTE_ENV")"
cp "$R2_ENV_FILE" "$R2_REMOTE_ENV"
SYNC_ARGS+=(--sync "$R2_REMOTE_ENV")

CLOUD_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  CLOUD_ARGS+=(--secure-cloud)
fi

DATA_CENTER_ARGS=()
if [[ -n "$DATA_CENTER_IDS" ]]; then
  DATA_CENTER_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi

RETENTION_ARGS=()
if [[ "$KEEP_POD" == "1" ]]; then
  RETENTION_ARGS+=(--keep-pod)
fi

OPTIONAL_TRAIN_ARGS=()
if [[ -n "$MAX_TRAIN_EVAL_EXAMPLES" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--max-train-eval-examples "$MAX_TRAIN_EVAL_EXAMPLES")
fi
if [[ -n "$MAX_EVAL_EXAMPLES" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--max-eval-examples "$MAX_EVAL_EXAMPLES")
fi
if [[ "$DISABLE_EARLY_STOPPING" == "1" ]]; then
  OPTIONAL_TRAIN_ARGS+=(--disable-early-stopping)
else
  OPTIONAL_TRAIN_ARGS+=(--early-stopping-patience "$EARLY_STOPPING_PATIENCE")
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$PWD" \
  --name "$RUNPOD_JOB_NAME" \
  --pod-name "$RUNPOD_JOB_NAME-$RUN_ID" \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${CLOUD_ARGS[@]}" \
  "${DATA_CENTER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds "$WAIT_SECONDS" \
  --ssh-wait-seconds "$SSH_WAIT_SECONDS" \
  --detached-remote \
  --remote-poll-seconds "$REMOTE_POLL_SECONDS" \
  --allow-existing-pods \
  "${RETENTION_ARGS[@]}" \
  "${SYNC_ARGS[@]}" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"
echo \"restore_cache prefix=$R2_CACHE_PREFIX tensor_cache=$TENSOR_CACHE\"
R2_ENV_FILE=\"$R2_REMOTE_ENV\" bash scripts/restore_r2_artifact.sh \"$R2_CACHE_PREFIX\" \"$TENSOR_CACHE\" | tee \"$OUTPUT_DIR/cache_restore_size.json\"
du -sh \"$TENSOR_CACHE\" | tee \"$OUTPUT_DIR/cache_size.txt\"
echo \"train_config assembly_spec=$ASSEMBLY_SPEC tensor_cache=$TENSOR_CACHE max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE eval_every=$EVAL_EVERY early_stopping_patience=$EARLY_STOPPING_PATIENCE disable_early_stopping=$DISABLE_EARLY_STOPPING\"
.venv/bin/python -u -m intrep.train_shogi_policy_value \
  --data-selection \"$DATA_SELECTION\" \
  --tensor-cache \"$TENSOR_CACHE\" \
  --checkpoint-path \"$OUTPUT_DIR/checkpoint.pt\" \
  --best-checkpoint-path \"$OUTPUT_DIR/best_checkpoint.pt\" \
  --metrics-path \"$OUTPUT_DIR/metrics.json\" \
  --max-steps \"$MAX_STEPS\" \
  --batch-size \"$BATCH_SIZE\" \
  --learning-rate \"$LEARNING_RATE\" \
  --weight-decay \"$WEIGHT_DECAY\" \
  --embedding-dim \"$EMBEDDING_DIM\" \
  --hidden-dim \"$HIDDEN_DIM\" \
  --num-heads \"$NUM_HEADS\" \
  --num-layers \"$NUM_LAYERS\" \
  --assembly-spec \"$ASSEMBLY_SPEC\" \
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
  ${OPTIONAL_TRAIN_ARGS[*]}" \
  "$@"
