#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bash scripts/setup_runpod.sh
.venv/bin/python -m pip install 'transformers==5.17.0' 'safetensors==0.8.0'
HF_HUB_DISABLE_PROGRESS_BARS=1 .venv/bin/python -m unittest \
  tests.test_shared_predictor tests.test_shared_prediction_sources tests.test_shared_prediction_evaluation \
  tests.test_shared_prediction_questions
