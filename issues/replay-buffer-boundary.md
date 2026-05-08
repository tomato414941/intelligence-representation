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
Replay Buffer should describe how training samples from stored experience.

Prefer a PyTorch-compatible shape:

- keep records close to source form in the Experience Store
- let Replay Buffer produce selected records or training examples
- use PyTorch `Dataset` for indexed samples
- use PyTorch `Sampler` when sampling weights or ordering matter
- avoid a generic multi-domain replay framework until a second concrete replay
  use case exists

The first implementation should likely be shogi-local and minimal. A clean KISS
entry point is a replay policy that creates or feeds a Training View while
recording the source mix, recency policy, duplicate policy, and maximum sample
count.

## Acceptance Criteria

This issue can close when a minimal Replay Buffer design has been implemented
or explicitly rejected.

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

## Related

- [`learning-data-flow-boundary.md`](learning-data-flow-boundary.md) tracks the
  broader supervised / self-supervised / reinforcement-learning data-flow
  boundary.
