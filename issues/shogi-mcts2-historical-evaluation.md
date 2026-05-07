# Shogi MCTS2 Historical Evaluation

Status: closed.

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

This issue is closed because MCTS2-derived runs are explicitly treated as
historical or smoke evidence unless a current evaluation policy names MCTS2.

Older MCTS2 results should not drive current model-quality decisions because:

- the arena MCTS final-selection value perspective bug has since been fixed,
- MCTS2 is too shallow to be the default practical search setting, and
- current battle evaluations should record the exact arena revision, policy
  mode, simulation count, opponent, and game count.

The current post-fix MCTS2/MCTS8 checks remain useful as contemporary evidence:
they show that the latest small policy+value model still does not beat
YaneuraOu nodes1, and that shallow MCTS is not yet improving play. That points
back to policy/value model strength rather than preserving old MCTS2
assumptions.
