#!/usr/bin/env bash
set -euo pipefail

# Build the shogi policy-plane tensor cache on a RunPod CPU Pod.
# The completed cache stays on the kept Pod; the local output directory receives
# only small run metadata and the cache manifest.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
RUNPOD_JOB_NAME=${RUNPOD_JOB_NAME:-intrep-shogi-policy-plane-cache}
RUNPOD_IMAGE=${RUNPOD_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}
CPU_FLAVOR_ID=${CPU_FLAVOR_ID:-cpu3g}
VCPU_COUNT=${VCPU_COUNT:-16}
JOBS=${JOBS:-"$VCPU_COUNT"}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-160}
VOLUME_SIZE=${VOLUME_SIZE:-0}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-720}
WAIT_SECONDS=${WAIT_SECONDS:-600}
SSH_WAIT_SECONDS=${SSH_WAIT_SECONDS:-180}
SECURE_CLOUD=${SECURE_CLOUD:-1}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

LOCAL_BUNDLE=${LOCAL_BUNDLE:-data/shogi/training-data-bundles/qhapaq-full}
DATA_SELECTION=${DATA_SELECTION:-"$LOCAL_BUNDLE/data-selection.json"}
CACHE_DIR=${CACHE_DIR:-"$LOCAL_BUNDLE/cache/policy-plane"}
SHARD_EXAMPLES=${SHARD_EXAMPLES:-10000}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-policy-plane-cache-$RUN_ID}

if [[ ! -x .venv/bin/python ]]; then
  echo ".venv/bin/python is required for local input discovery" >&2
  exit 1
fi

mapfile -t INPUT_FILES < <(
  .venv/bin/python -m intrep.problems.shogi_policy_value.training_inputs \
    --data-selection "$DATA_SELECTION"
)

SYNC_ARGS=()
for input_file in "${INPUT_FILES[@]}"; do
  SYNC_ARGS+=(--sync "$input_file")
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
  --pod-name "$RUNPOD_JOB_NAME-$RUN_ID" \
  --compute-type CPU \
  --image "$RUNPOD_IMAGE" \
  --cpu-flavor-id "$CPU_FLAVOR_ID" \
  --vcpu-count "$VCPU_COUNT" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  "${CLOUD_ARGS[@]}" \
  "${DATA_CENTER_ARGS[@]}" \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds "$WAIT_SECONDS" \
  --ssh-wait-seconds "$SSH_WAIT_SECONDS" \
  --allow-existing-pods \
  --keep-pod \
  --sync src \
  --sync pyproject.toml \
  --sync uv.lock \
  --sync README.md \
  --sync AGENTS.md \
  --sync scripts/setup_runpod_cpu.sh \
  --sync scripts/build_shogi_policy_value_tensor_cache_parallel.py \
  "${SYNC_ARGS[@]}" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod_cpu.sh' \
  --output "$OUTPUT_DIR" \
  --timings-output "$OUTPUT_DIR/runpod_timings.json" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"
echo \"cache_build_config data_selection=$DATA_SELECTION cache_dir=$CACHE_DIR shard_examples=$SHARD_EXAMPLES jobs=$JOBS output_space=policy_plane\"
.venv/bin/python -u scripts/build_shogi_policy_value_tensor_cache_parallel.py \
  --data-selection \"$DATA_SELECTION\" \
  --out \"$CACHE_DIR\" \
  --output-space policy_plane \
  --shard-examples \"$SHARD_EXAMPLES\" \
  --jobs \"$JOBS\" \
  --resume | tee \"$OUTPUT_DIR/cache_build_summary.json\"
cp \"$CACHE_DIR/manifest.json\" \"$OUTPUT_DIR/cache_manifest.json\"
du -sh \"$CACHE_DIR\" | tee \"$OUTPUT_DIR/cache_size.txt\"
printf '%s\n' \"$CACHE_DIR\" > \"$OUTPUT_DIR/remote_cache_path.txt\"" \
  "$@"
