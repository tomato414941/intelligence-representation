# Shogi RL Artifact Boundary doc placement

## Problem

`docs/learning-boundaries.md` is the project-level learning-boundary document.
It currently contains a `Shogi RL Artifact Boundary` section with shogi-specific
runtime and repository-boundary details.

That content may belong in `docs/shogi/learning-boundaries.md` instead, because
it discusses:

- the `intelligence-representation` / `shogi-arena-agent` boundary
- shogi game generation runtime
- player-vs-player match entrypoints
- RunPod match wrappers
- shogi game-record artifacts

Keeping too much shogi-specific operational detail in the project-level
document can make the top-level document less clearly about generic learning
concepts.

## Desired Direction

Keep `docs/learning-boundaries.md` focused on general learning concepts such as
source records, data selection, runs, datasets, objectives, and recursive
execution.

Move or summarize the shogi-specific artifact-boundary material under
`docs/shogi/`, with only a short pointer from the project-level document if
needed. Use a separate shogi document if `docs/shogi/learning-boundaries.md` is
not ready to absorb the material.

## Close When

- The project-level learning-boundary document no longer carries detailed
  shogi-specific runtime ownership rules.
- The shogi-specific rules remain documented under `docs/shogi/`.
- The two documents do not duplicate the same boundary rules.
