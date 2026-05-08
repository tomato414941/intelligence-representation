# Replay Buffer Boundary

Status: open.

## Issue

The project should introduce Replay Buffer as a learning-time sampling layer,
not as another name for Experience Store.

The current shogi flow has durable storage and fixed views, but no explicit
Replay Buffer layer:

```text
Experience Store
  -> Training View
  -> Data Selection
  -> PyTorch Dataset
  -> Training Loop
```

This works for supervised-style fixed datasets, but it leaves RL replay behavior
implicit in view creation.

The missing replay questions are:

- how much recent vs old experience to train on
- how to mix teacher-vs-teacher, teacher-vs-model, and model-vs-model games
- how to make duplicate positions visible or controlled
- whether weak historical model experience should still affect training
- how to sample enough diverse positions without manually rebuilding views

Those are not storage problems. They are replay-policy and sampling problems.

## Why It Matters

If Experience Store and Training View remain the only concepts, training can
drift toward fixed supervised datasets. That is simple, but it may be too rigid
for reinforcement-learning-style improvement where new experience is generated,
replayed, mixed, and partially forgotten over time.

The project should decide whether it needs a Replay Buffer or Sampler layer
between stored experience and training batches.

## Direction

Do not rename Experience Store to Replay Buffer.

Experience Store should remain the durable source of generated experience.
Replay Buffer should describe how training samples from reusable experience.
It should not require Experience Store as its only input; Experience Store is
one possible source of game-record JSONL, alongside run outputs, Qhapaq-derived
records, teacher-only records, or other generated records.

Prefer a PyTorch-compatible shape:

- keep records close to source form in the Experience Store
- let Replay Buffer produce selected records or training examples
- use PyTorch `Dataset` for indexed samples
- use PyTorch `Sampler` when sampling weights or ordering matter
- avoid a generic multi-domain replay framework until a second concrete replay
  use case exists

The first implementation is shogi-local and minimal:

- `scripts/create_shogi_replay_view.py`
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
