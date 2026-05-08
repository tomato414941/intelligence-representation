# Source Selection Generalization Boundary

Status: open. Priority: low.

## Issue

`source selection` is a general-looking concept, but the only concrete
implementation is currently shogi-local:

- `src/intrep/worlds/shogi/source_selection.py`
- `scripts/create_shogi_training_view_from_sources.py`

The current implementation depends on shogi-specific details:

- `ShogiGameRecord`
- `black_actor` / `white_actor`
- actor-pair strings
- shogi game-record JSONL
- shogi Training View output

## Current Policy

Keep source selection shogi-local for now.

Do not create a shared source-selection package only because the concept looks
general. Generalize only after another concrete use case shows the same boundary
with different record types, group keys, outputs, and training targets.

## Trigger

Revisit when a second concrete source-selection use case exists, for example
grid interaction records, tool-use traces, browser interaction records, or
another generated-experience source.

## Non-Goals

- move shogi source-selection code into `core/`
- redesign the shogi Training View format
- define a generic source-selection abstraction now
