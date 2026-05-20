#!/usr/bin/env bash
set -euo pipefail

# Thin RunPod transport wrapper. It owns remote execution mechanics only; the
# command passed through REMOTE_COMMAND owns all training or workload semantics.

cd "$(dirname "$0")/../.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
RUNPOD_JOB_NAME=${RUNPOD_JOB_NAME:-intrep-run-command}
RUNPOD_TEMPLATE_ID=${RUNPOD_TEMPLATE_ID:-runpod-torch-v280}
GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
WAIT_SECONDS=${WAIT_SECONDS:-600}
SSH_WAIT_SECONDS=${SSH_WAIT_SECONDS:-180}
SECURE_CLOUD=${SECURE_CLOUD:-1}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
OUTPUT_DIR=${OUTPUT_DIR:-runs/runpod-command}
SETUP_COMMAND=${SETUP_COMMAND:-'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh'}

if [[ -z "${REMOTE_COMMAND:-}" ]]; then
  echo "REMOTE_COMMAND is required" >&2
  exit 2
fi

SYNC_ARGS=()
for sync_path in ${SYNC_PATHS:-}; do
  SYNC_ARGS+=(--sync "$sync_path")
done

CLOUD_ARGS=()
if [[ "$SECURE_CLOUD" == "1" ]]; then
  CLOUD_ARGS+=(--secure-cloud)
fi

DATA_CENTER_ARGS=()
if [[ -n "$DATA_CENTER_IDS" ]]; then
  DATA_CENTER_ARGS+=(--data-center-ids "$DATA_CENTER_IDS")
fi

python3 "$RUNPOD_JOB" \
  --repo-root "$PWD" \
  --name "$RUNPOD_JOB_NAME" \
  --template-id "$RUNPOD_TEMPLATE_ID" \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${CLOUD_ARGS[@]}" \
  "${DATA_CENTER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds "$WAIT_SECONDS" \
  --ssh-wait-seconds "$SSH_WAIT_SECONDS" \
  --allow-existing-pods \
  --sync src \
  --sync tests \
  --sync pyproject.toml \
  --sync uv.lock \
  --sync README.md \
  --sync AGENTS.md \
  --sync scripts/setup_runpod.sh \
  "${SYNC_ARGS[@]}" \
  --setup-command "$SETUP_COMMAND" \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"; $REMOTE_COMMAND" \
  "$@"
