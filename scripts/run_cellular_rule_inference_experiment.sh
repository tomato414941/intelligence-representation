#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${RUN_DIR:?Set RUN_DIR to a new experiment output directory}"
PYTHON=${PYTHON:-.venv/bin/python}
DEVICE=${DEVICE:-cuda}
TRAINING_STEPS=${TRAINING_STEPS:-6000}
MODEL_SEED=${MODEL_SEED:-31}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

"$PYTHON" -u -m intrep.train_cellular_rule_inference \
  --run-dir "$RUN_DIR" --max-steps "$TRAINING_STEPS" --model-seed "$MODEL_SEED" --device "$DEVICE"
"$PYTHON" -u -m intrep.problems.cellular_rule_inference.evaluate \
  --checkpoint "$RUN_DIR/checkpoint.pt" --output "$RUN_DIR/validation.json" \
  --rule-count 32 --queries-per-rule 4 --device "$DEVICE"
"$PYTHON" -u -m intrep.problems.cellular_rule_inference.evaluate \
  --checkpoint "$RUN_DIR/checkpoint.pt" --output "$RUN_DIR/test.json" \
  --rule-count 64 --queries-per-rule 8 --rule-seed 19101 --data-seed 23001 \
  --exclude-rules-from "$RUN_DIR/validation.json" --device "$DEVICE"
