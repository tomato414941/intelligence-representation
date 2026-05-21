#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi

"$PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pip install "numpy<2" "python-shogi>=1.1.1" "tokenizers>=0.23.1" "torch>=2.2" "zstandard>=0.23"
.venv/bin/python - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
PY
