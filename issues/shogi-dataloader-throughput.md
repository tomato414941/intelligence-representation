# Shogi DataLoader Throughput

## Issue

Shogi move-choice training uses PyTorch `Dataset` and `DataLoader`, but the data
path is still likely too Python-heavy for efficient GPU training.

## Current State

| Area | Current behavior |
| --- | --- |
| dataset | `ShogiMoveChoiceDataset` wraps loaded Python examples |
| loader | `DataLoader(..., shuffle=True)` with default worker settings |
| workers | `num_workers=0` by default |
| transfer | tensors are moved to the target device inside the training loop |
| item conversion | `__getitem__` converts SFEN and legal moves to tensors per sample |

## Problem

For small models, GPU compute can finish quickly and then wait for CPU-side
example conversion or single-worker batch construction. This makes GPU runtime
hard to interpret and can waste cloud GPU time.

## Candidate Direction

Measure first with `--log-every`. If the model is data-bound, consider exposing
minimal DataLoader settings such as `num_workers` and `pin_memory`, then only
move toward tensorized or binary caches if the simple PyTorch settings are not
enough.
