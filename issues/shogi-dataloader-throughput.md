# Shogi DataLoader Throughput

Status: open.

## Issue

Shogi move-choice training uses PyTorch `Dataset` and `DataLoader`, but the data
path is still likely too Python-heavy for efficient GPU training.

## Current State

| Area | Current behavior |
| --- | --- |
| dataset | `ShogiMoveChoiceDataset` wraps loaded Python examples |
| loader | PyTorch `DataLoader` is used for train, train-eval, and held-out eval |
| workers | `num_workers` is configurable from the training CLI; the RunPod script defaults to `0` |
| pinned memory | `pin_memory` is configurable from the training CLI |
| transfer | tensors are moved to the target device inside the training loop |
| item conversion | `__getitem__` converts SFEN and legal moves to tensors per sample |

## Problem

For small models, GPU compute can still finish quickly and then wait for
CPU-side example conversion or batch construction.

The first KISS fix exposed DataLoader settings through the training CLI, but
the full Qhapaq JSONL cache showed a second problem: multiple DataLoader workers
can increase CPU RAM pressure because the dataset is a large Python object list.
On a 46 GB RunPod, a 2000-step run with `num_workers=4` reached 350 steps and
then the pod stopped responding over SSH. A local probe with 100k examples also
showed worker processes increasing proportional set size as they touched the
object cache.

## Candidate Direction

The training CLI exposes minimal DataLoader settings:

- `--num-workers`
- `--pin-memory`

Current operational default:

| Setting | Default | Reason |
| --- | --- | --- |
| `NUM_WORKERS` | `0` | Avoid private-copy RAM growth from the full Python-object cache. |
| `DATA_CENTER_IDS` | unset, but use `EU-RO-1` for longer RunPod baselines | `EU-RO-1` completed the 2000-step workers0 baseline; a US-CA-2 host failed during this workstream. |

Before increasing `NUM_WORKERS`, either:

- measure CPU RAM pressure on a full-cache run, or
- replace the Python-object dataset with a tensorized or binary cache that can
  be shared by workers without large private-copy growth.

Move toward tensorized or binary caches if worker-free throughput becomes the
main bottleneck.
