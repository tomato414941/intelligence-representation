# Shogi World Package Boundary

Status: closed.

Resolution: closed via acceptance criterion (b). The learning-data lifecycle
code (`training_data_bundle`, `game_split`, `experience_stats`,
`generated_record_archive`) moved out of the world package into
`intrep.experience.shogi` as part of the 2026-06-10 source-package
reorganization (`worlds/` + `sources/` + `experience/`). Board and move
encoding had already moved to `representation/inputs` and
`representation/outputs` on 2026-05-20. The world package
(`intrep.worlds.shogi`) now holds world records, rules, and engine
interfaces only.

## Issue

`src/intrep/worlds/shogi/` owns both shogi world representation and some shogi
learning-data lifecycle code.

Current responsibilities include:

- source / record representation: `game_record.py`, `game_replay.py`, `kif_io.py`
- board and move encoding: `position_encoding.py`, `move_encoding.py`
- learning-data lifecycle: `training_data_bundle.py`, `experience_stats.py`,
  `game_split.py`
- USI info inspection: `info_stats.py`, `inspect_usi_info.py`
- a few package modules with CLI-style `main()` functions

This is currently workable, but the boundary is broad.

## Why It Matters

If this keeps growing unchecked:

- `worlds/shogi/` may become the default home for every shogi-shaped feature
- data lifecycle code may hide learning assumptions inside world code
- CLI entrypoints may creep back into package modules

Resolved concrete smells:

- `worlds/shogi` no longer imports from `intrep.problems`.
- `kif_io.py` stops at `ShogiGameRecord`.
- actor-pair counts live in `experience_stats.py`.

## Current Policy

Leave the package as-is for now.

Do not split the package only for architectural neatness. Revisit when the shogi
learning-data lifecycle code grows again or another world needs the same
lifecycle.

## Candidate Direction

If this is revisited, first decide whether Training Data Bundle behavior is
intentionally shogi-world lifecycle code or should move to a shared
learning-data boundary.

## Acceptance Criteria

This issue can close when the project either:

- accepts that `worlds/shogi/` owns shogi learning-data lifecycle code for now,
  or
- moves that lifecycle code to a clearer boundary because there is a concrete
  second use case or concrete maintenance pain.

## Non-Goals

- move files immediately
- introduce generic world data lifecycle abstractions
- redesign Training Data Bundle or Replay Buffer
