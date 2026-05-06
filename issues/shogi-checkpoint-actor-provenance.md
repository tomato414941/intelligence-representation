# Shogi Checkpoint Actor Provenance

Status: open.

## Issue

Shogi Experience Store records can include checkpoint actors, but checkpoint
provenance is still fragile once many model generations and search settings are
mixed.

Current summaries track actor pairs and checkpoint actor counts, including
checkpoint path, policy, and simulations. That is enough for small experiments,
but it is not yet a durable way to reason about model generations.

## Why It Matters

Checkpoint-generated experience is not all equivalent.

Examples:

- checkpoint A with direct policy
- checkpoint A with MCTS2
- checkpoint A with MCTS8
- checkpoint B with MCTS8
- checkpoint B with MCTS16

These produce different position distributions and different policy/value
signals. If the store cannot clearly explain which checkpoint generation and
search settings produced each slice of experience, later Training Views may
mix weak, stale, or incompatible data without that being obvious.

The risk grows when local `runs/.../checkpoint.pt` paths are deleted, renamed,
or become hard to interpret.

## Scope

- Decide what checkpoint identity should be recorded for generated shogi
  experience.
- Decide whether a short generation name, checkpoint path, git commit, run name,
  model config, policy, and search settings are enough.
- Decide what should appear in Experience Store manifest/history summaries.
- Decide what Training View needs in order to include, exclude, or cap
  checkpoint-generated experience by generation or search settings.

## Non-Goals

- Do not introduce a broad model registry before a concrete need exists.
- Do not require long-lived storage for every local checkpoint as part of this
  issue.
- Do not solve source-mix selection here; that belongs in
  `shogi-training-view-source-mix.md`.

## Acceptance Criteria

This issue can close when checkpoint-generated shogi experience has a clear,
durable provenance policy, and Experience Store / Training View metadata can
explain which checkpoint generation and search settings a training slice came
from.
