# Shogi Outdated RL Assumptions

Status: open.

## Issue

Some shogi learning assumptions are now outdated relative to the current RL and
Experience Store direction.

Outdated assumptions or stale paths to review:

- Qhapaq-specific RunPod/cost/cache paths that predate the current Training
  View and Experience Store flow.
- run output directories as the practical unit of training-data management.
- winner-derived value as a sufficient value teacher.
- very shallow MCTS, such as MCTS2, as a primary playing-strength signal.
- checkpoint actor generation, policy, and search settings being hard to trace
  from accumulated experience data.

## Why It Matters

The current direction is to grow a persistent shogi Experience Store, create
intentional Training Views, and improve policy/value from generated games and
teacher annotations.

Keeping old assumptions around makes the next experiments harder to interpret.
The project can appear to be improving dataset eval while playing strength does
not improve, or it can train on whatever happened to be generated most recently
instead of a deliberate experience mix.

## Acceptance Criteria

This issue can close when the outdated assumptions or stale paths above are
either removed, replaced, or explicitly kept with a current reason.

The resolution should not add a broad generic RL framework. Keep the cleanup
grounded in the current shogi Experience Store, Training View, and battle
evaluation flow.
