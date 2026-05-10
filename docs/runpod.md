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

## Shogi Training

Current entrypoint:

```sh
scripts/runpod_train_shogi_policy_value.sh
```

This script uses the repository-local `scripts/runpod/run_job.py` helper.
It must not depend on RunPod helpers from another project.

Local RunPod credentials and SSH key paths are provided at runtime through CLI
arguments or environment variables: `RUNPOD_API_KEY` or `RUNPOD_API_KEY_FILE`,
`RUNPOD_SSH_KEY`, `RUNPOD_SSH_PUBLIC_KEY`, and optionally `RUNPODCTL`.

This entrypoint is policy-value specific. Generalizing shogi RunPod training is
tracked in `issues/runpod-shogi-training-entrypoint-generalization.md`.

## Runtime Choices

For full-cache shogi runs, keep `NUM_WORKERS=0` unless CPU RAM behavior has
been measured on the target cache and Pod size. The JSONL cache is loaded as a
large Python object list, and workers can increase RAM pressure.

For longer baselines, prefer `DATA_CENTER_IDS=EU-RO-1`; `US-CA-2` has had SSH
stability failures during shogi training.

The script uses disposable compute and currently avoids RunPod network volumes.
Network-volume use is tracked separately in
`issues/runpod-network-volume-revisit.md`.

## Images

Repository-local RunPod job helper defaults:

```text
runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
```

Allowed CUDA versions in the job helper:

```text
12.8, 12.9, 13.0
```

Some RTX 4090 hosts cannot start CUDA 12.8 images when their NVIDIA driver is
too old. This is a container startup failure, not a training-code or search
parameter problem.

Known manual fallback for profiling and smoke work:

```text
runpod-torch-v240
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
```

This fallback was used successfully for MCTS large-batch profiling on
2026-05-10. It is not yet the standard path for
`scripts/runpod_train_shogi_policy_value.sh`.

## Records

- Cost, runtime, memory, and throughput: `docs/compute-costs.md`
- Inference latency and output throughput: `docs/inference-performance.md`
