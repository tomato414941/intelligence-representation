# Shogi RL Assumption Review

Status: open.

## Issue

Some shogi learning assumptions and older experiment paths need review as the
project moves toward RL-style generated experience and Training Views.

Assumptions or stale paths still to review:

- MCTS2 is no longer the current default search setting, but older shogi
  experiments and Experience Store entries still include MCTS2-derived data.
  Treat those runs as historical evidence or smoke data unless a current
  evaluation policy explicitly chooses MCTS2.

## Why It Matters

The current direction is to grow a persistent shogi Experience Store, create
intentional Training Views, and improve policy/value from generated games and
teacher annotations.

Keeping unreviewed assumptions around makes the next experiments harder to interpret.
The project can appear to be improving dataset eval while playing strength does
not improve, or it can train on whatever happened to be generated most recently
instead of a deliberate experience mix.

## Acceptance Criteria

This issue can close when the remaining MCTS2 note is either removed, split into
a concrete battle-evaluation issue, or explicitly kept as historical-context
documentation.

The resolution should not add a broad generic RL framework. Keep the cleanup
grounded in the current shogi Experience Store, Training View, and battle
evaluation flow.
