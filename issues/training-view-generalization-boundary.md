# Training View Generalization Boundary

Status: open. Priority: low.

## Issue

`Training View` is currently implemented only for shogi:

- `scripts/create_shogi_training_view.py`
- `src/intrep/worlds/shogi/training_view.py`

The underlying idea may be broader than shogi: materialize a fixed training /
evaluation input from source records, selected records, target policy, and
manifest metadata. However, the current implementation depends on shogi-specific
details:

- `ShogiGameRecord`
- `black_actor` / `white_actor`
- actor-pair strings
- shogi game-record JSONL
- shogi Training View output

## Current Policy

Keep Training View implementation shogi-local for now.

Do not create a shared Training View abstraction only because the concept looks
general. Generalize only after another concrete use case shows the same boundary
with different record types, outputs, target construction, and training
interface.

## Trigger

Revisit when a second concrete Training View-like use case exists, for example
image records with multiple objectives, text corpora with fixed subsets, grid
interaction records, tool-use traces, or browser interaction records.

## Non-Goals

- move shogi training-view code into `core/`
- redesign the shogi Training View format
- define a generic Training View abstraction now
