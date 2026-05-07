#!/usr/bin/env bash
set -euo pipefail

# Use RunPod as disposable compute.
# Keep this on container disk; network volumes caused pod readiness failures.

cd "$(dirname "$0")/.."

DATASET_DEFINITION=${DATASET_DEFINITION:-data/shogi/datasets/current/dataset.json}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-shogi-move-choice}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
# Keep worker count at zero for the current JSONL/Python-object cache. Worker
# processes gradually private-copy the large example list and can make 46 GB
# pods unresponsive during longer full-cache runs. Revisit after a tensorized or
# binary cache exists.
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
EMBEDDING_DIM=${EMBEDDING_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-6}
# Optional RunPod data-center pin. EU-RO-1 has completed the 2000-step
# full-cache baseline; a US-CA-2 host failed with SSH timeout and CUDA init
# errors during the same workstream.
DATA_CENTER_IDS=${DATA_CENTER_IDS:-}

if [[ ! -f "$DATASET_DEFINITION" ]]; then
  echo "dataset definition not found: $DATASET_DEFINITION" >&2
  exit 1
fi

mapfile -t DATASET_FILES < <(
  .venv/bin/python - "$DATASET_DEFINITION" <<'PY'
import json
import sys
from pathlib import Path

definition_path = Path(sys.argv[1])
payload = json.loads(definition_path.read_text(encoding="utf-8"))
paths = {definition_path}
for key in ("train_sources", "eval_sources"):
    for source in payload.get(key, []):
        if source.get("kind") != "game_records_jsonl":
            continue
        source_path = Path(source["path"])
        if not source_path.is_absolute():
            source_path = definition_path.parent / source_path
        paths.add(source_path)
for path in sorted(paths):
    if not path.exists():
        raise SystemExit(f"dataset source not found: {path}")
    print(path)
PY
)
SYNC_ARGS=()
for dataset_file in "${DATASET_FILES[@]}"; do
  SYNC_ARGS+=(--sync "$dataset_file")
done

python3 /home/dev/projects/llm/scripts/runpod/run_once.py \
  --repo-root "$PWD" \
  --name intrep-shogi-move-choice \
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
  --no-cuda-smoke \
  --sync scripts/setup_runpod.sh \
  "${SYNC_ARGS[@]}" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"
echo \"run_config max_steps=$MAX_STEPS batch_size=$BATCH_SIZE learning_rate=$LEARNING_RATE policy_loss_weight=$POLICY_LOSS_WEIGHT value_loss_weight=$VALUE_LOSS_WEIGHT embedding_dim=$EMBEDDING_DIM hidden_dim=$HIDDEN_DIM num_heads=$NUM_HEADS num_layers=$NUM_LAYERS num_workers=$NUM_WORKERS max_train_eval_examples=$MAX_TRAIN_EVAL_EXAMPLES max_eval_examples=$MAX_EVAL_EXAMPLES checkpoint_every=$CHECKPOINT_EVERY metrics_every=$METRICS_EVERY keep_last_n_checkpoints=$KEEP_LAST_N_CHECKPOINTS eval_every=$EVAL_EVERY\"
.venv/bin/python -u -m intrep.train_shogi_move_choice \
  --dataset-definition \"$DATASET_DEFINITION\" \
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
  --keep-last-n-checkpoints \"$KEEP_LAST_N_CHECKPOINTS\"" \
  "$@"
