# Shogi MCTS2 Historical Evaluation

Status: open.

## Issue

MCTS2 is no longer the current default search setting, but older shogi
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

This issue can close when MCTS2-derived runs are either removed from active
decision-making, split into a concrete battle-evaluation issue, or explicitly
kept as historical-context documentation.

The resolution should not add a broad generic RL framework. Keep the cleanup
grounded in the current shogi Experience Store, Training View, and battle
evaluation flow.
