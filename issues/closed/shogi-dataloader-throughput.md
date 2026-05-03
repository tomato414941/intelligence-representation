# Shogi DataLoader Throughput

Status: closed.
Resolution: resolved for the current KISS throughput path.

## Issue

Shogi move-choice training uses PyTorch `Dataset` and `DataLoader`, but the data
path is still likely too Python-heavy for efficient GPU training.

## Current State

| Area | Current behavior |
| --- | --- |
| dataset | `ShogiMoveChoiceDataset` wraps loaded Python examples |
| loader | PyTorch `DataLoader` is used for train, train-eval, and held-out eval |
| workers | `num_workers` is configurable from the training CLI; the RunPod script defaults to `0` for memory safety |
| pinned memory | `pin_memory` is configurable from the training CLI |
| transfer | tensors are moved to the target device inside the training loop |
| item conversion | `__getitem__` converts SFEN and legal moves to tensors per sample |

## Problem

For small models, GPU compute can still finish quickly and then wait for
CPU-side example conversion or batch construction.

The first KISS fix exposed DataLoader settings through the training CLI. That
solved the immediate throughput-control problem: runs can now choose worker
count and pinned memory without editing code.

The full Qhapaq JSONL cache later exposed a separate memory/stability problem:
multiple DataLoader workers can increase CPU RAM pressure because the dataset is
a large Python object list. Track that separately in
[`shogi-full-cache-memory.md`](shogi-full-cache-memory.md).

## Candidate Direction

The training CLI exposes minimal DataLoader settings:

- `--num-workers`
- `--pin-memory`

Current operational default for full-cache RunPod jobs:

| Setting | Default | Reason |
| --- | --- | --- |
| `NUM_WORKERS` | `0` | Avoid private-copy RAM growth from the full Python-object cache until the memory issue is fixed. |
| `DATA_CENTER_IDS` | unset, but use `EU-RO-1` for longer RunPod baselines | `EU-RO-1` completed the 2000-step workers0 baseline; a US-CA-2 host failed during this workstream. |

Keep this issue closed unless worker-free throughput becomes the main bottleneck
after the memory issue is handled.
