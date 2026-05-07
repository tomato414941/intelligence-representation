# Experience Store Generalization Boundary

Status: open.

## Issue

Experience Store is currently implemented only for shogi, but the concept is
not inherently shogi-specific.

Generated or collected experience may also appear in future grid, text, image,
agent, or other world/task work. In those cases, keeping Experience Store only
inside shogi could lead to duplicated store concepts or unclear project
boundaries.

The current shogi-local placement is acceptable for KISS/YAGNI: shogi is the
only concrete implementation, and extracting a shared abstraction now would be
premature.

## Why It Matters

Experience Store means durable storage for generated or collected experience.
That responsibility is broader than shogi game records, even if the current
schema is shogi-specific.

The project should avoid both bad outcomes:

- prematurely creating a generic Experience Store framework from only shogi
- letting shogi-specific store assumptions become the hidden default for all
  future generated experience

## Initial Policy

Keep the current implementation shogi-local until a second concrete area needs
durable generated experience.

When a second implementation exists, compare the two concrete stores and extract
only the shared boundary that is actually needed.

## Acceptance Criteria

This issue can close when one of the following is true:

- a second concrete Experience Store use case exists and a minimal shared
  boundary is extracted from both implementations
- the project decides Experience Store should intentionally remain
  world/task-specific and documents why
- the term Experience Store is replaced by a clearer project-wide concept

## Non-Goals

- redesign shogi Experience Store now
- introduce a generic ExperienceStore class from shogi alone
- solve Training View lifecycle boundaries
- implement Replay Buffer or sampling policy
