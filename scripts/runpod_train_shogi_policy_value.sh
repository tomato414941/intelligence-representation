#!/usr/bin/env bash
set -euo pipefail

# Use RunPod as disposable compute.
# Project-specific RunPod operating notes live in docs/runpod.md.
# Keep this on container disk for the current KISS flow; network-volume
# reevaluation is tracked in issues/runpod-network-volume-revisit.md.

cd "$(dirname "$0")/.."

DATA_SELECTION=${DATA_SELECTION:-data/shogi/training-data-bundles/current/data-selection.json}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-shogi-policy-value}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
# Keep worker count at zero for the current JSONL/Python-object cache. See
# docs/runpod.md before increasing this for full-cache runs.
NUM_WORKERS=${NUM_WORKERS:-0}
LEARNING_RATE=${LEARNING_RATE:-0.0005}
POLICY_LOSS_WEIGHT=${POLICY_LOSS_WEIGHT:-1.0}
VALUE_LOSS_WEIGHT=${VALUE_LOSS_WEIGHT:-0.0}
MAX_TRAIN_EVAL_EXAMPLES=${MAX_TRAIN_EVAL_EXAMPLES:-4096}
MAX_EVAL_EXAMPLES=${MAX_EVAL_EXAMPLES:-4096}
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
# Optional RunPod data-center pin. See docs/runpod.md before long baselines.
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

if [[ ! -f "$DATA_SELECTION" ]]; then
  echo "data selection not found: $DATA_SELECTION" >&2
  exit 1
fi

mapfile -t DATA_SELECTION_FILES < <(
  .venv/bin/python - "$DATA_SELECTION" <<'PY'
import json
import sys
from pathlib import Path

selection_path = Path(sys.argv[1])
payload = json.loads(selection_path.read_text(encoding="utf-8"))
paths = {selection_path}
for key in ("train_sources", "eval_sources", "analysis_sources"):
    for source in payload.get(key, []):
        if source.get("kind") not in {"game_records_jsonl", "shogi_engine_analysis_jsonl"}:
            continue
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = selection_path.parent / source_path
        paths.add(source_path)
for path in sorted(paths):
    if not path.exists():
        raise SystemExit(f"data selection source not found: {path}")
    print(path)
PY
)
SYNC_ARGS=()
for selection_file in "${DATA_SELECTION_FILES[@]}"; do
  SYNC_ARGS+=(--sync "$selection_file")
done

python3 scripts/runpod/run_once.py \
  --repo-root "$PWD" \
  --name intrep-shogi-policy-value \
  --secure-cloud \
  --gpu-type "NVIDIA GeForce RTX 4090" \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --allowed-cuda-version 12.8 \
  --allowed-cuda-version 12.9 \
  --allowed-cuda-version 13.0 \
  --container-disk-size 80 \
  --volume-size 0 \
  --remote-dir /root/intrep \
  --mem 32 \
  --vcpu 8 \
  ${DATA_CENTER_IDS:+--data-center-ids "$DATA_CENTER_IDS"} \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 600 \
  --ssh-wait-seconds 180 \
  --allow-existing-pods \
  --sync scripts/setup_runpod.sh \
  "${SYNC_ARGS[@]}" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"
echo \"run_config max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT embedding_dim=$EMBEDDING_DIM hidden_dim=$HIDDEN_DIM num_heads=$NUM_HEADS num_layers=$NUM_LAYERS num_workers=$NUM_WORKERS max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES checkpoint_every=$CHECKPOINT_EVERY metrics_every=$METRICS_EVERY keep_last_n_checkpoints=$KEEP_LAST_N_CHECKPOINTS eval_every=$EVAL_EVERY early_stopping_patience=$EARLY_STOPPING_PATIENCE\"
TRAIN_ARGS=()
if [[ -n \"$EARLY_STOPPING_PATIENCE\" ]]; then
  TRAIN_ARGS+=(--early-stopping-patience \"$EARLY_STOPPING_PATIENCE\")
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
  --policy-loss-weight \"$POLICY_LOSS_WEIGHT\" \
  --value-loss-weight \"$VALUE_LOSS_WEIGHT\" \
  --device cuda \
  --log-every 50 \
  --eval-every \"$EVAL_EVERY\" \
  --num-workers \"$NUM_WORKERS\" \
  --pin-memory \
  --max-train-eval-examples \"$MAX_TRAIN_EVAL_EXAMPLES\" \
  --max-eval-examples \"$MAX_EVAL_EXAMPLES\" \
  --checkpoint-every \"$CHECKPOINT_EVERY\" \
  --metrics-every \"$METRICS_EVERY\" \
  --keep-last-n-checkpoints \"$KEEP_LAST_N_CHECKPOINTS\" \
  \"\${TRAIN_ARGS[@]}\"" \
  "$@"
