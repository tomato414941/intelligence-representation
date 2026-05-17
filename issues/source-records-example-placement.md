# Source Records example placement

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
