# Replay Buffer vs Experience Store

Status: open.

## Issue

The current shogi data flow may have solved persistence before solving the
learning-time sampling problem.

Experience Store is useful for durable generated experience, but recent
questions are more about Replay Buffer behavior:

- how much recent vs old experience to train on
- how to mix teacher-vs-teacher, teacher-vs-model, and model-vs-model games
- how to make duplicate positions visible or controlled
- whether weak historical model experience should still affect training
- how to sample enough diverse positions without manually rebuilding views

Those are not primarily storage problems. They are sampling and replay-policy
problems.

## Why It Matters

If Experience Store and Training View remain the only concepts, training can
drift toward fixed supervised datasets. That is simple, but it may be too rigid
for reinforcement-learning-style improvement where new experience is generated,
replayed, mixed, and partially forgotten over time.

The project should decide whether it needs a Replay Buffer or Sampler layer
between stored experience and training batches.

## Initial Policy

Do not rename Experience Store to Replay Buffer.

Experience Store should remain the durable source of generated experience.
Replay Buffer, if introduced, should describe how training samples from stored
experience. Prefer PyTorch-compatible `Dataset` / `Sampler` concepts over a
custom framework unless a concrete RL loop needs more.

## Acceptance Criteria

This issue can close when the project has decided one of the following:

- Experience Store + fixed Training View is enough for the current RL loop
- a minimal Replay Buffer / Sampler layer is needed and implemented
- the same need is covered by a simpler Training View sampling policy

The decision should explain what owns source mix, duplicate handling, recency,
and maximum sample count.

## Non-Goals

- implement prioritized replay immediately
- introduce a generic multi-domain replay framework
- change ShogiGameRecord schema
- remove Experience Store persistence
