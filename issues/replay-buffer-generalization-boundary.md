# Replay Buffer Generalization Boundary

Status: open. Priority: low.

## Issue

Replay selection is currently implemented only for shogi:

- `src/intrep/worlds/shogi/replay.py`
- `scripts/create_shogi_replay_view.py`

The concept is broader than shogi, but the current implementation depends on
shogi-specific records and fields:

- `ShogiGameRecord`
- `black_actor` / `white_actor`
- actor-pair strings
- shogi game-record JSONL
- shogi Training View output

## Current Policy

Keep Replay Buffer / replay selection shogi-local for now.

This is intentional KISS. The project has only one concrete replay use case, and
generic replay abstractions would likely encode shogi assumptions too early.

## Trigger

Revisit when a second concrete replay use case exists, for example grid
interaction replay, tool-use replay, browser interaction replay, or another
generated-experience source that needs sampling beyond static Data Selection.

At that point, compare the concrete implementations and extract only the shared
boundary that is actually needed.

## Non-Goals

- introduce a generic `ReplayBuffer` class now
- move shogi replay code into `core/`
- rename Experience Store
- implement prioritized or online replay
