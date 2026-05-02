#!/usr/bin/env bash
set -euo pipefail

# Use RunPod as disposable compute.
# Keep this on container disk; network volumes caused pod readiness failures.

cd "$(dirname "$0")/.."

TRAIN_CACHE_ZST=${TRAIN_CACHE_ZST:-runs/shogi/qhapaq-train-move-choice-examples.jsonl.zst}
EVAL_CACHE_ZST=${EVAL_CACHE_ZST:-runs/shogi/qhapaq-eval-move-choice-examples.jsonl.zst}
OUTPUT_DIR=${OUTPUT_DIR:-runs/shogi/runpod-qhapaq-split-b512-steps5000}
MAX_STEPS=${MAX_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-512}
MAX_RUNTIME_MINUTES=${MAX_RUNTIME_MINUTES:-420}
NUM_WORKERS=${NUM_WORKERS:-4}
EMBEDDING_DIM=${EMBEDDING_DIM:-256}
HIDDEN_DIM=${HIDDEN_DIM:-1024}
NUM_HEADS=${NUM_HEADS:-8}
NUM_LAYERS=${NUM_LAYERS:-6}

if [[ ! -f "$TRAIN_CACHE_ZST" ]]; then
  echo "compressed train cache not found: $TRAIN_CACHE_ZST" >&2
  exit 1
fi
if [[ ! -f "$EVAL_CACHE_ZST" ]]; then
  echo "compressed eval cache not found: $EVAL_CACHE_ZST" >&2
  exit 1
fi

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
  --max-cost 1.0 \
  --max-runtime-minutes "$MAX_RUNTIME_MINUTES" \
  --wait-seconds 600 \
  --ssh-wait-seconds 180 \
  --allow-existing-pods \
  --no-cuda-smoke \
  --sync scripts/setup_runpod.sh \
  --sync "$TRAIN_CACHE_ZST" \
  --sync "$EVAL_CACHE_ZST" \
  --setup-command 'cd "$REMOTE_DIR"; bash scripts/setup_runpod.sh' \
  --output "$OUTPUT_DIR" \
  --remote "set -euo pipefail; cd \"\$REMOTE_DIR\"; mkdir -p \"$OUTPUT_DIR\"; .venv/bin/python - <<'PY'
import zstandard as zstd
from pathlib import Path

for name in ('$TRAIN_CACHE_ZST', '$EVAL_CACHE_ZST'):
    src = Path(name)
    dst = src.with_suffix('')
    print(f'decompressing {src} -> {dst}', flush=True)
    with src.open('rb') as f, dst.open('wb') as out:
        zstd.ZstdDecompressor().copy_stream(f, out)
    print(f'decompressed_bytes={dst.stat().st_size}', flush=True)
PY
.venv/bin/python -u -m intrep.train_shogi_move_choice \
  --train-examples-jsonl \"${TRAIN_CACHE_ZST%.zst}\" \
  --eval-examples-jsonl \"${EVAL_CACHE_ZST%.zst}\" \
  --checkpoint-path \"$OUTPUT_DIR/checkpoint.pt\" \
  --metrics-path \"$OUTPUT_DIR/metrics.json\" \
  --max-steps \"$MAX_STEPS\" \
  --batch-size \"$BATCH_SIZE\" \
  --learning-rate 0.003 \
  --weight-decay 0.01 \
  --embedding-dim \"$EMBEDDING_DIM\" \
  --hidden-dim \"$HIDDEN_DIM\" \
  --num-heads \"$NUM_HEADS\" \
  --num-layers \"$NUM_LAYERS\" \
  --value-loss-weight 0.2 \
  --device cuda \
  --log-every 50 \
  --num-workers \"$NUM_WORKERS\" \
  --pin-memory \
  --max-train-eval-examples 4096"
