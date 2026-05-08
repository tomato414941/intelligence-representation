# Replay Buffer Boundary

Status: open.

## Issue

The project should distinguish Online Experience Replay from Offline Experience
Reuse.

Online Experience Replay is a training-time RL method backed by a dynamic
Replay Buffer. Offline Experience Reuse is pre-training data selection over
previously collected experience records.

The current shogi flow can read fixed game-record JSONL sources and build fixed
Training Views, but it has no explicit Replay Buffer layer:

```text
game-record sources
  -> Training View
  -> Data Selection
  -> PyTorch Dataset
  -> Training Loop
```

Those game-record sources may come from an Experience Store, run outputs,
teacher-only records, Qhapaq-derived records, or other generated record sets.
The current implementation is Offline Experience Reuse: it selects records once
and writes a fixed Training View before training starts.

The current fixed-view flow works for supervised-style datasets, but it leaves
Online Experience Replay behavior unimplemented.

The missing replay questions are:

- how much recent vs old experience to train on
- how to mix teacher-vs-teacher, teacher-vs-model, and model-vs-model games
- how to make duplicate positions visible or controlled
- whether weak historical model experience should still affect training
- how to sample enough diverse positions without manually rebuilding views

Those are not storage problems. They are sampling-policy problems.

## Why It Matters

If source records and Training View remain the only concepts, training can drift
toward fixed supervised datasets. That is simple, but it may be too rigid for
reinforcement-learning-style improvement where new experience is generated,
replayed, mixed, and partially forgotten over time.

The project should decide whether it needs a Replay Buffer or Sampler layer
between stored experience and training batches.

## Direction

Do not rename Experience Store to Replay Buffer.

Experience Store should remain one durable source of generated experience.
Replay Buffer should describe training-time sampling from reusable experience
records. It must not require Experience Store as its input.

Prefer a PyTorch-compatible shape:

- keep records close to source form in source storage
- let Offline Experience Reuse produce fixed selected records or training
  examples
- reserve Replay Buffer for training-time sampling
- use PyTorch `Dataset` for indexed samples
- use PyTorch `Sampler` when sampling weights or ordering matter
- avoid a generic multi-domain replay framework until a second concrete replay
  use case exists

The first implementation is shogi-local and minimal:

- `scripts/create_shogi_training_view_from_sources.py`
- reusable behavior lives in `src/intrep/worlds/shogi/source_selection.py`
- accepts one or more `--train-games` JSONL sources
- accepts fixed `--eval-games`
- supports `--max-train-games`, `--max-eval-games`, and
  `--actor-pair-ratio`
- outputs a fixed Training View plus `data-selection.json`
- records available counts, selected counts, actor-pair mix, duplicate stats,
  and source paths in `manifest.json`

This is a source-selection training-view builder, not an online RL replay buffer.

## Acceptance Criteria

This issue can close when the minimal source-selection training-view builder is judged sufficient
for the current RL loop, or when the project decides an online Replay Buffer /
PyTorch Sampler layer is now needed.

The decision must explain what owns:

- source mix
- duplicate handling
- recency
- maximum sample count
- relationship to PyTorch `Dataset` / `Sampler`
- relationship to fixed Training Views

## Non-Goals

- implement prioritized replay immediately
- introduce a generic multi-domain replay framework
- change ShogiGameRecord schema
- remove Experience Store persistence
- implement online replay sampling inside the training loop

## Related

- [`learning-data-flow-boundary.md`](learning-data-flow-boundary.md) tracks the
  broader supervised / self-supervised / reinforcement-learning data-flow
  boundary.
