# Replay Buffer Boundary

Status: open.

## Issue

The project should treat replay as a learning-time sampling layer over reusable
experience records, not as another name for Experience Store.

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
The replay concern is the sampling policy over these sources, not the storage
location that produced them.

The current fixed-view flow works for supervised-style datasets, but it leaves
RL replay behavior implicit in view creation.

The missing replay questions are:

- how much recent vs old experience to train on
- how to mix teacher-vs-teacher, teacher-vs-model, and model-vs-model games
- how to make duplicate positions visible or controlled
- whether weak historical model experience should still affect training
- how to sample enough diverse positions without manually rebuilding views

Those are not storage problems. They are replay-policy and sampling problems.

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
Replay Buffer should describe how training samples from reusable experience
records. It must not require Experience Store as its input.

Prefer a PyTorch-compatible shape:

- keep records close to source form in source storage
- let Replay Buffer produce selected records or training examples
- use PyTorch `Dataset` for indexed samples
- use PyTorch `Sampler` when sampling weights or ordering matter
- avoid a generic multi-domain replay framework until a second concrete replay
  use case exists

The first implementation is shogi-local and minimal:

- `scripts/create_shogi_replay_view.py`
- reusable behavior lives in `src/intrep/worlds/shogi/replay.py`
- accepts one or more `--train-games` JSONL sources
- accepts fixed `--eval-games`
- supports `--max-train-games`, `--max-eval-games`, and
  `--actor-pair-ratio`
- outputs a fixed Training View plus `data-selection.json`
- records available counts, selected counts, actor-pair mix, duplicate stats,
  and source paths in `manifest.json`

This is a replay-view builder, not an online RL replay buffer.

## Acceptance Criteria

This issue can close when the minimal replay-view builder is judged sufficient
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
