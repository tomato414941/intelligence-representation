# Experience Store Generalization Boundary

Status: open.

## Issue

Experience Store is currently implemented only for shogi, but the concept is
not inherently shogi-specific.

Generated or collected experience may also appear in future grid, text, image,
agent, or other world/task work. In those cases, keeping Experience Store only
inside shogi could lead to duplicated store concepts or unclear project
boundaries.

The related lifecycle is also currently shogi-local:

- append generated records to an Experience Store
- create a fixed Training Data Bundle
- train through Data Selection or a problem-specific training input definition

Replay Buffer is now a separate training-time component under `intrep.learning`.
That does not change this issue: Experience Store is still durable source
storage, not online replay sampling.

The current shogi-local placement is acceptable for KISS/YAGNI: shogi is the
only concrete implementation, and extracting a shared abstraction now would be
premature.

## Why It Matters

Experience Store means durable storage for generated or collected experience.
That responsibility is broader than shogi game records, even if the current
schema is shogi-specific.

The same caution applies to the Experience Store -> Training Data Bundle lifecycle.
It may become useful beyond shogi, but extracting a shared abstraction from one
implementation would be premature.

The project should avoid both bad outcomes:

- prematurely creating a generic Experience Store framework from only shogi
- letting shogi-specific store assumptions become the hidden default for all
  future generated experience

## Initial Policy

Keep the current implementation shogi-local until a second concrete area needs
durable generated experience.

When a second implementation exists, compare the two concrete stores and extract
only the shared boundary that is actually needed. If the second area also needs
fixed Training Data Bundles, include that lifecycle in the comparison.

## Acceptance Criteria

This issue can close when one of the following is true:

- a second concrete Experience Store use case exists and a minimal shared
  boundary is extracted from both implementations
- the project decides Experience Store should intentionally remain
  world/problem-specific and documents why
- the term Experience Store is replaced by a clearer project-wide concept

## Non-Goals

- redesign shogi Experience Store now
- introduce a generic ExperienceStore class from shogi alone
- redesign Replay Buffer or sampling policy

## Related

The narrower Training Data Bundle lifecycle issue was merged into this issue:

- [`closed/experience-store-training-data-bundle-boundary.md`](closed/experience-store-training-data-bundle-boundary.md)
