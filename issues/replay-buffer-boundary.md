# Replay Buffer Boundary

Status: open.

## Issue

The project needs a clear boundary for Replay Buffer before adding one.

Online Experience Replay is a training-time RL method backed by a dynamic
Replay Buffer. It is different from Offline Experience Reuse, which selects
previously collected records once and builds a fixed Training Data Bundle before
training starts.

A Replay Buffer should mean training-time sampling from reusable experience, not
durable source storage and not one-time Training Data Bundle construction.

## Why It Matters

Without this boundary, future RL work can mix up four different responsibilities:

- durable experience storage
- fixed source selection / Training Data Bundle construction
- PyTorch `Dataset` / `Sampler`
- training-time replay sampling

## Direction

Do not rename Experience Store to Replay Buffer.

A Replay Buffer should own training-time sampling questions such as:

- recent vs old experience
- replacement or capacity policy
- random, weighted, or prioritized sampling
- duplicate experience handling during sampling
- interaction with PyTorch `Dataset` / `Sampler` / `DataLoader`

It should not require Experience Store as its only input. Experience Store is
one possible durable source of experience records.

## Acceptance Criteria

This issue can close when Replay Buffer has a concrete design or implementation
that explains what owns:

- training-time sampling
- capacity / replacement
- recency or priority
- duplicate handling
- relationship to PyTorch `Dataset` / `Sampler`
- relationship to Experience Store and fixed Training Data Bundles

## Non-Goals

- change ShogiGameRecord schema
- rename Experience Store
- redefine Offline Experience Reuse

## Related

- [`learning-data-flow-boundary.md`](learning-data-flow-boundary.md) tracks the
  broader supervised / self-supervised / reinforcement-learning data-flow
  boundary.
