# Source Records example placement

Status: closed

## Problem

`docs/learning-data-boundaries.md` has a project-level source-side material section. It
should stay focused on general learning-boundary principles.

Shogi-specific examples such as game records, generated games, traces, engine
analysis records, or policy/value examples are useful, but placing them inside
the project-level source-side material section can blur the document's scope.

## Desired Direction

Keep `docs/learning-data-boundaries.md` focused on learning data boundaries.

Put shogi-specific source-record examples under `docs/shogi/learning-boundaries.md`
or another shogi-specific document if examples are needed.

## Close When

- The project-level source-side material section remains domain-neutral.
- Any shogi-specific source-record examples are documented only in `docs/shogi/`.
- The distinction between project-level principle and shogi-specific example is
  clear.

## Resolution

2026-05-20:

- `docs/learning-data-boundaries.md` keeps the source-side material section
  domain-neutral.
- Shogi-specific source-record examples live under
  `docs/shogi/learning-boundaries.md`.
- No additional documentation move is needed for this issue.
