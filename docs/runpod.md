# RunPod

This document is the project-specific RunPod operations note for
`intelligence-representation`.

## Setup

This repo's RunPod setup is designed to use the PyTorch/CUDA stack already
provided by the selected RunPod template.

Do not run `uv sync` on RunPod, because it can replace the template's system
PyTorch with a wheel whose CUDA build does not match the host NVIDIA driver.
Use:

```sh
./scripts/setup_runpod.sh
```

`setup_runpod.sh` verifies system `torch` and CUDA, creates `.venv` with
`--system-site-packages`, installs this repo with `pip install -e . --no-deps`,
and installs non-torch runtime dependencies explicitly.

For torchvision jobs, run `./scripts/setup_runpod_vision.sh` after
`setup_runpod.sh` and provide a torchvision wheel matching the selected image.

## RunPod Command Runner

Current entrypoint:

```sh
scripts/runpod/run_command.sh
```

This script is a thin transport wrapper around the shared
`../runpod-job-runner/scripts/run_job.py` helper. It owns RunPod pod lifecycle,
repository sync, setup, output collection, and timing capture. It does not own
training settings.

Pass the actual workload through `REMOTE_COMMAND`. For shogi policy/value
training, the canonical training interface is still
`python -m intrep.train_shogi_policy_value`.

Example:

```sh
SYNC_PATHS="data/shogi/training-data-bundles/qhapaq-full/data-selection.json data/shogi/tensor-caches/qhapaq-full/cache.pt" \
OUTPUT_DIR="runs/shogi/full-001" \
REMOTE_COMMAND='.venv/bin/python -u -m intrep.train_shogi_policy_value \
  --data-selection data/shogi/training-data-bundles/qhapaq-full/data-selection.json \
  --tensor-cache data/shogi/tensor-caches/qhapaq-full/cache.pt \
  --checkpoint-path runs/shogi/full-001/checkpoint.pt \
  --best-checkpoint-path runs/shogi/full-001/best_checkpoint.pt \
  --metrics-path runs/shogi/full-001/metrics.json \
  --policy-output shogi_policy_plane_policy_output \
  --max-steps 5000 \
  --batch-size 512 \
  --device cuda \
  --num-workers 0' \
scripts/runpod/run_command.sh
```

Local RunPod credentials and SSH key paths are provided at runtime through CLI
arguments or environment variables: `RUNPOD_API_KEY` or `RUNPOD_API_KEY_FILE`,
`RUNPOD_SSH_KEY`, `RUNPOD_SSH_PUBLIC_KEY`, and optionally `RUNPODCTL`.

## Shogi Checkpoint

RunPod shogi runs use the promoted d256 checkpoint by default:

```text
models/d256-h1024-heads8-l6-shogi/checkpoint.pt
```

The d32 checkpoint is for local smoke tests. Do not use it for RunPod training
or arena evaluation unless the run is explicitly a d32 smoke.

RunPod shogi self-play generation uses the `cshogi` board backend by default.
The `python-shogi` backend is kept as a compatibility option.

## Runtime Choices

For full-cache shogi runs, pass `--num-workers 0` unless CPU RAM behavior has
been measured on the target cache and Pod size. The JSONL cache is loaded as a
large Python object list, and workers can increase RAM pressure.

Use `DATA_CENTER_IDS` only when a run needs an explicit data-center pin. Leave
it unset by default so scheduling can adapt to current RunPod availability.

The script uses disposable compute and currently avoids RunPod network volumes.
Network-volume use is tracked separately in
`issues/runpod-network-volume-revisit.md`.

## GPU Selection

This project currently assumes NVIDIA CUDA GPUs on the RunPod PyTorch 2.8
template. Do not use AMD GPUs for the current RunPod shogi jobs.

Prefer cost-sensitive CUDA GPUs first:

- RTX 3090
- RTX A5000
- A40

Use faster or larger CUDA GPUs when cheaper GPUs are unavailable:

- RTX A6000
- RTX 4090
- RTX 5090

Use datacenter CUDA GPUs only when the run justifies the cost:

- A100
- H100
- H200

## Images

Repository-local RunPod job helper default template:

```text
runpod-torch-v280
Runpod Pytorch 2.8.0
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

The helper uses the official RunPod PyTorch template by default. Use `--image`
only for explicit fallback work.

The helper does not set `allowedCudaVersions` by default. If a job needs to
constrain host CUDA compatibility, pass `--allowed-cuda-version` explicitly.

Previously used CUDA 12.8 image:

```text
runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
```

Some RTX 4090 hosts cannot start CUDA 12.8 images when their NVIDIA driver is
too old. This is a container startup failure, not a training-code or search
parameter problem.

The `runpod-torch-v280` template was verified with RTX 5090 on 2026-05-10:
`torch 2.8.0+cu128`, CUDA available, and the repository setup smoke completed.

## Records

- Cost, runtime, memory, and throughput: `docs/compute-costs.md`
- Shogi play inference latency and output throughput:
  `docs/shogi/play-inference-performance.md`
