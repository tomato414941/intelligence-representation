# Shogi World Package Boundary

Status: open.

## Issue

`src/intrep/worlds/shogi/` now owns more than pure shogi world representation.

Current responsibilities include:

- source / record representation: `game_record.py`, `game_replay.py`, `kif_io.py`
- board and move encoding: `position_encoding.py`, `move_encoding.py`
- learning-data lifecycle: `experience_store.py`, `training_data_bundle.py`,
  `replay.py`, `experience_stats.py`, `game_split.py`
- USI info inspection: `info_stats.py`, `inspect_usi_info.py`
- a few package modules with CLI-style `main()` functions

This is currently workable, but the boundary is broad. The package is becoming
both a shogi world package and a shogi learning-data lifecycle package.

## Why It Matters

If this keeps growing unchecked:

- `worlds/shogi/` may become the default home for every shogi-shaped feature
- world representation may depend on problem-specific policy/value concerns
- data lifecycle code may hide learning assumptions inside world code
- CLI entrypoints may creep back into package modules

The previous clearest smell was `kif_io.py` importing from
`intrep.problems.shogi_policy_value`. That dependency has been removed:
`kif_io.py` now stops at `ShogiGameRecord`, and KIF-to-problem-example
conversion lives under `problems/shogi_policy_value/data.py`.

## Current Policy

Leave the package as-is for now. The recent move from scripts into importable
package code was still an improvement.

Do not split the package only for architectural neatness. Revisit when editing
one of these areas substantially.

## Candidate Direction

Possible future split:

- keep shogi rules, records, replay validation, KIF/USI record IO, and
  board/move encodings under `worlds/shogi/`
- keep Experience Store / Training Data Bundle / Replay code either under
  `worlds/shogi/` with explicit lifecycle naming, or move it only if another
  concrete world needs the same lifecycle
- keep package modules importable; put CLI wrappers in scripts or proper module
  entrypoints

## Acceptance Criteria

- Decide whether `worlds/shogi/` intentionally owns shogi learning-data
  lifecycle code.
- Keep `worlds/shogi/` free of problem-specific imports.
- Decide whether package-local `main()` functions are acceptable here.

## Non-Goals

- move files immediately
- introduce generic world data lifecycle abstractions
- redesign Experience Store, Training Data Bundle, or Replay Buffer
