#!/usr/bin/env bash
set -euo pipefail

# Profile DataLoader worker counts for shogi policy/value tensor-cache training
# on one disposable RunPod job.

cd "$(dirname "$0")/.."

RUNPOD_RUNNER_ROOT=${RUNPOD_RUNNER_ROOT:-"$PWD/../runpod-job-runner"}
RUNPOD_JOB=${RUNPOD_JOB:-"$RUNPOD_RUNNER_ROOT/scripts/run_job.py"}
DATA_SELECTION=${DATA_SELECTION:-data/shogi/training-data-bundles/current/data-selection.json}
TENSOR_CACHE=${TENSOR_CACHE:-}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-shogi-policy-value-dataloader-profile}
WORKER_COUNTS=${WORKER_COUNTS:-0 2 4 8}
MAX_STEPS=${MAX_STEPS:-500}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
GPU_TYPE=${GPU_TYPE:-NVIDIA RTX A5000}
CONTAINER_DISK_SIZE=${CONTAINER_DISK_SIZE:-80}
VOLUME_SIZE=${VOLUME_SIZE:-0}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-0.0}
ASSEMBLY_SPEC=${ASSEMBLY_SPEC:-shogi_policy_value_position_transformer_legal_move_attention}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-16384}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-16384}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-500}
METRICS_EVERY=${METRICS_EVERY:-500}
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-1}
EVAL_EVERY=${EVAL_EVERY:-500}
EMBEDDING_DIM=${EMBEDDING_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-6}
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

if [[ -z "$TENSOR_CACHE" ]]; then
  echo "TENSOR_CACHE is required for DataLoader profiling" >&2
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

TRAINING_INPUT_ARGS=(--data-selection "$DATA_SELECTION" --tensor-cache "$TENSOR_CACHE")
mapfile -t TRAINING_INPUT_FILES < <(
  .venv/bin/python -m intrep.problems.shogi_policy_value.training_inputs "${TRAINING_INPUT_ARGS[@]}"
)
SYNC_ARGS=()
for input_file in "${TRAINING_INPUT_FILES[@]}"; do
  SYNC_ARGS+=(--sync "$input_file")
done

python3 "$RUNPOD_JOB" \
  --repo-root "$PWD" \
  --name intrep-shogi-policy-value-dataloader-profile \
  --template-id runpod-torch-v280 \
  --gpu-type "$GPU_TYPE" \
  --container-disk-size "$CONTAINER_DISK_SIZE" \
  --volume-size "$VOLUME_SIZE" \
  --secure-cloud \
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
echo \"profile_config worker_counts=$WORKER_COUNTS max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT embedding_dim=$EMBEDDING_DIM hidden_dim=$HIDDEN_DIM num_heads=$NUM_HEADS num_layers=$NUM_LAYERS assembly_spec=$ASSEMBLY_SPEC max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES tensor_cache=$TENSOR_CACHE\"
for worker_count in $WORKER_COUNTS; do
  case_dir=\"$OUTPUT_DIR/workers-\$worker_count\"
  mkdir -p \"\$case_dir\"
  echo \"profile_case num_workers=\$worker_count output=\$case_dir\"
  .venv/bin/python -u -m intrep.train_shogi_policy_value \
    --data-selection \"$DATA_SELECTION\" \
    --tensor-cache \"$TENSOR_CACHE\" \
    --checkpoint-path \"\$case_dir/checkpoint.pt\" \
    --best-checkpoint-path \"\$case_dir/best_checkpoint.pt\" \
    --metrics-path \"\$case_dir/metrics.json\" \
    --max-steps \"$MAX_STEPS\" \
    --batch-size \"$BATCH_SIZE\" \
    --learning-rate \"$LEARNING_RATE\" \
    --weight-decay 0.01 \
    --embedding-dim \"$EMBEDDING_DIM\" \
    --hidden-dim \"$HIDDEN_DIM\" \
    --num-heads \"$NUM_HEADS\" \
    --num-layers \"$NUM_LAYERS\" \
    --assembly-spec \"$ASSEMBLY_SPEC\" \
    --policy-loss-weight \"$POLICY_LOSS_WEIGHT\" \
    --value-loss-weight \"$VALUE_LOSS_WEIGHT\" \
    --device cuda \
    --log-every 50 \
    --eval-every \"$EVAL_EVERY\" \
    --num-workers \"\$worker_count\" \
    --pin-memory \
    --max-train-eval-examples \"$MAX_TRAIN_EVAL_EXAMPLES\" \
    --max-eval-examples \"$MAX_EVAL_EXAMPLES\" \
    --checkpoint-every \"$CHECKPOINT_EVERY\" \
    --metrics-every \"$METRICS_EVERY\" \
    --keep-last-n-checkpoints \"$KEEP_LAST_N_CHECKPOINTS\"
done" \
  "$@"
