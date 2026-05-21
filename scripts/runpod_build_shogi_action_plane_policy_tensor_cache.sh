#!/usr/bin/env bash
set -euo pipefail

# Build a shogi action-plane policy tensor cache on a RunPod CPU Pod.
#
# CACHE_RETENTION=pod keeps the Pod after success so the cache can be used there.
# CACHE_RETENTION=measure deletes the Pod after success and keeps only run metadata.
# Container disk is not persistent: deleting the Pod deletes the remote cache.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)}
RUNPOD_JOB_NAME=${RUNPOD_JOB_NAME:-intrep-shogi-action-plane-policy-cache}
RUNPOD_IMAGE=${RUNPOD_IMAGE:-runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404}
CPU_FLAVOR_ID=${CPU_FLAVOR_ID:-cpu5c}
VCPU_COUNT=${VCPU_COUNT:-4}
JOBS=${JOBS:-"$VCPU_COUNT"}
ASSEMBLY_SPEC=${ASSEMBLY_SPEC:-}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-60}
VOLUME_SIZE=${VOLUME_SIZE:-0}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-720}
WAIT_SECONDS=${WAIT_SECONDS:-600}
SSH_WAIT_SECONDS=${SSH_WAIT_SECONDS:-180}
SECURE_CLOUD=${SECURE_CLOUD:-0}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}
CACHE_RETENTION=${CACHE_RETENTION:-pod}
MAX_TRAIN_EXAMPLES=${MAX_TRAIN_EXAMPLES:-}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-}

LOCAL_BUNDLE=${LOCAL_BUNDLE:-data/shogi/training-data-bundles/qhapaq-full}
DATA_SELECTION=${DATA_SELECTION:-"$LOCAL_BUNDLE/data-selection.json"}
SHARD_EXAMPLES=${SHARD_EXAMPLES:-100000}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-action-plane-policy-cache-$RUN_ID}

if [[ -z "$ASSEMBLY_SPEC" ]]; then
  echo "ASSEMBLY_SPEC is required for tensor cache construction" >&2
  exit 1
fi
if [[ "$CACHE_RETENTION" != "pod" && "$CACHE_RETENTION" != "measure" ]]; then
  echo "CACHE_RETENTION must be pod or measure" >&2
  exit 1
fi
CACHE_DIR=${CACHE_DIR:-"$LOCAL_BUNDLE/cache/$ASSEMBLY_SPEC"}

if [[ ! -x .venv/bin/python ]]; then
  echo ".venv/bin/python is required for local input discovery" >&2
  exit 1
fi

LIMITED_DATA_SELECTION=
cleanup_limited_data_selection() {
  if [[ -n "$LIMITED_DATA_SELECTION" ]]; then
    rm -f "$LIMITED_DATA_SELECTION"
  fi
}
trap cleanup_limited_data_selection EXIT

if [[ -n "$MAX_TRAIN_EXAMPLES" || -n "$MAX_EVAL_EXAMPLES" ]]; then
  LIMITED_DATA_SELECTION="$(dirname "$DATA_SELECTION")/data-selection-limited-$RUN_ID.json"
  .venv/bin/python - "$DATA_SELECTION" "$LIMITED_DATA_SELECTION" "$MAX_TRAIN_EXAMPLES" "$MAX_EVAL_EXAMPLES" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
max_train_examples = sys.argv[3]
max_eval_examples = sys.argv[4]
payload = json.loads(source_path.read_text(encoding="utf-8"))
payload["name"] = f"{payload['name']}-limited"
if max_train_examples:
    payload["train_sources"][0]["max_examples"] = int(max_train_examples)
if max_eval_examples:
    payload["eval_sources"][0]["max_examples"] = int(max_eval_examples)
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  DATA_SELECTION="$LIMITED_DATA_SELECTION"
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

RETENTION_ARGS=()
if [[ "$CACHE_RETENTION" == "pod" ]]; then
  RETENTION_ARGS+=(--keep-pod)
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
  "${RETENTION_ARGS[@]}" \
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
echo \"cache_build_config data_selection=$DATA_SELECTION cache_dir=$CACHE_DIR shard_examples=$SHARD_EXAMPLES jobs=$JOBS assembly_spec=$ASSEMBLY_SPEC\"
.venv/bin/python -u scripts/build_shogi_policy_value_tensor_cache_parallel.py \
  --data-selection \"$DATA_SELECTION\" \
  --out \"$CACHE_DIR\" \
  --assembly-spec \"$ASSEMBLY_SPEC\" \
  --shard-examples \"$SHARD_EXAMPLES\" \
  --jobs \"$JOBS\" \
  --resume \
  --summary-output \"$OUTPUT_DIR/cache_build_summary.json\" | tee \"$OUTPUT_DIR/cache_build_events.jsonl\"
cp \"$CACHE_DIR/manifest.json\" \"$OUTPUT_DIR/cache_manifest.json\"
du -sh \"$CACHE_DIR\" | tee \"$OUTPUT_DIR/cache_size.txt\"
printf '%s\n' \"$CACHE_DIR\" > \"$OUTPUT_DIR/remote_cache_path.txt\"" \
  "$@"
