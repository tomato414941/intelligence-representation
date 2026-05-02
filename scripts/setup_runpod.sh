#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi

"$PYTHON" - <<'PY'
try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("RunPod setup expects an official PyTorch template with system torch installed.") from error

if not torch.cuda.is_available():
    raise SystemExit("RunPod setup expects an official PyTorch template with CUDA available.")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
PY

"$PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install "numpy<2" "python-shogi>=1.1.1" "tokenizers>=0.23.1"
