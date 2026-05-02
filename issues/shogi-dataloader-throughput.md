# Shogi DataLoader Throughput

Status: resolved for the current KISS path.

## Issue

Shogi move-choice training uses PyTorch `Dataset` and `DataLoader`, but the data
path is still likely too Python-heavy for efficient GPU training.

## Current State

| Area | Current behavior |
| --- | --- |
| dataset | `ShogiMoveChoiceDataset` wraps loaded Python examples |
| loader | PyTorch `DataLoader` is used for train, train-eval, and held-out eval |
| workers | `num_workers` is configurable from the training CLI |
| pinned memory | `pin_memory` is configurable from the training CLI |
| transfer | tensors are moved to the target device inside the training loop |
| item conversion | `__getitem__` converts SFEN and legal moves to tensors per sample |

## Problem

For small models, GPU compute can still finish quickly and then wait for CPU-side
example conversion or batch construction. The immediate issue was that the
training path could not change basic PyTorch loader settings without editing
code.

## Candidate Direction

The training CLI now exposes minimal DataLoader settings:

- `--num-workers`
- `--pin-memory`

Keep this as the current stopping point. Move toward tensorized or binary caches
only after measured RunPod runs show that these simple PyTorch settings are not
enough.
