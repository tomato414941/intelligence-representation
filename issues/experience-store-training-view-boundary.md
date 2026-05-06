# Experience Store Training View Boundary

Status: open.

## Issue

Experience Store and Training View lifecycle support exists only for shogi.

That may be correct for the first implementation, but the project is expected
to learn from many kinds of generated experience. If grid, image, text, or other
tasks also need appendable experience and fixed training snapshots, keeping the
workflow only inside shogi scripts will become a project-structure problem.

## Why It Matters

The current shogi lifecycle is:

- append generated ShogiGameRecord data to an Experience Store
- create an immutable Training View
- train from the Training View through a dataset definition

This lifecycle may also apply to future RL, self-supervised, or generated-data
work outside shogi. Without an explicit boundary, the project can drift into
either of two bad states:

- duplicate task-specific store/view implementations
- premature generic abstractions shaped only by shogi

## Initial Policy

Do not introduce a shared ExperienceStore or TrainingView abstraction from shogi
alone.

Keep the shogi implementation concrete until a second task needs the same
lifecycle. When that happens, compare both concrete implementations and extract
only the shared boundary that is actually needed.

## Acceptance Criteria

This issue can close when one of the following is true:

- a second concrete task uses the same lifecycle and the project introduces a
  minimal shared boundary from both implementations
- the project decides that store/view lifecycle should intentionally remain
  task-specific and documents that decision with examples

## Non-Goals

- optimize shogi Training View loading performance
- add tensorized shogi caches
- introduce a generic store/view framework before a second concrete use case
