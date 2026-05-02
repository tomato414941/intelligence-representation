#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi

# Keep RunPod's official system torch intact. Install torchvision only when a
# wheel matching the selected RunPod image is explicitly provided.
"$PYTHON" - <<'PY'
try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("RunPod vision setup expects system torch to be installed.") from error

if not torch.cuda.is_available():
    raise SystemExit("RunPod vision setup expects CUDA to be available.")

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0))
PY

if "$PYTHON" - <<'PY'
try:
    import torchvision
except ModuleNotFoundError:
    raise SystemExit(1)

print("torchvision", torchvision.__version__)
PY
then
  exit 0
fi

if [[ -z "${TORCHVISION_PACKAGE:-}" || -z "${TORCHVISION_INDEX_URL:-}" ]]; then
  cat >&2 <<'EOF'
torchvision is not installed.

Set TORCHVISION_PACKAGE and TORCHVISION_INDEX_URL explicitly for the selected
RunPod PyTorch image, for example:

  TORCHVISION_PACKAGE='torchvision==0.24.1+cu128' \
  TORCHVISION_INDEX_URL='https://download.pytorch.org/whl/cu128' \
  ./scripts/setup_runpod_vision.sh

This script intentionally does not infer the torch/torchvision/CUDA compatibility
matrix. The selected RunPod image is the source of truth.
EOF
  exit 1
fi

"$PYTHON" -m pip install "$TORCHVISION_PACKAGE" --index-url "$TORCHVISION_INDEX_URL" --no-deps

"$PYTHON" - <<'PY'
import torchvision

print("torchvision", torchvision.__version__)
PY
