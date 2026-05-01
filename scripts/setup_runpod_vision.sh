#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  PYTHON=python3
fi

"$PYTHON" - <<'PY'
import re
import subprocess
import sys

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("RunPod vision setup expects system torch to be installed.") from error

if not torch.cuda.is_available():
    raise SystemExit("RunPod vision setup expects CUDA to be available.")

match = re.match(r"^(\d+\.\d+\.\d+)", torch.__version__)
if match is None:
    raise SystemExit(f"Unsupported torch version string: {torch.__version__}")

torchvision_versions = {
    "2.2.0": "0.17.0",
    "2.2.1": "0.17.1",
    "2.2.2": "0.17.2",
    "2.3.0": "0.18.0",
    "2.3.1": "0.18.1",
    "2.4.0": "0.19.0",
    "2.4.1": "0.19.1",
    "2.5.0": "0.20.0",
    "2.5.1": "0.20.1",
}
cuda_indexes = {
    "12.1": "cu121",
    "12.4": "cu124",
}

torch_version = match.group(1)
cuda_version = torch.version.cuda
torchvision_version = torchvision_versions.get(torch_version)
cuda_tag = cuda_indexes.get(cuda_version or "")
if torchvision_version is None or cuda_tag is None:
    raise SystemExit(
        "Unsupported RunPod torch/CUDA pair: "
        f"torch={torch.__version__} cuda={cuda_version}. "
        "Install a matching torchvision wheel manually from the PyTorch wheel index."
    )

package = f"torchvision=={torchvision_version}+{cuda_tag}"
index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
subprocess.run(
    [sys.executable, "-m", "pip", "install", package, "--index-url", index_url, "--no-deps"],
    check=True,
)

import torchvision

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
PY
