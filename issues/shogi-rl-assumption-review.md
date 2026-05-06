# Shogi RL Assumption Review

Status: open.

## Issue

Some shogi learning assumptions and older experiment paths need review as the
project moves toward RL-style generated experience and Training Views.

Assumptions or stale paths still to review:

- very shallow MCTS, such as MCTS2, as a primary playing-strength signal.

## Why It Matters

The current direction is to grow a persistent shogi Experience Store, create
intentional Training Views, and improve policy/value from generated games and
teacher annotations.

Keeping unreviewed assumptions around makes the next experiments harder to interpret.
The project can appear to be improving dataset eval while playing strength does
not improve, or it can train on whatever happened to be generated most recently
instead of a deliberate experience mix.

## Acceptance Criteria

This issue can close when the assumptions or stale paths above are either
removed, replaced, split into concrete issues, or explicitly kept with a current
reason.

The resolution should not add a broad generic RL framework. Keep the cleanup
grounded in the current shogi Experience Store, Training View, and battle
evaluation flow.
