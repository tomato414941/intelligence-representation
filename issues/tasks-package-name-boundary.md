# Tasks Package Name Boundary

Status: open.

## Issue

`tasks/` remains as a real package name, but the project now treats `task` as an
informal word rather than a precise boundary.

This is different from the closed `task-package-responsibility-boundary.md`
issue. That issue narrowed what a task package may own. This issue is only about
whether the package name `tasks/` still fits.

## Current State

`docs/glossary.md` says `Task Package` is a historical package name for a
problem-oriented model surface.

`docs/architecture.md` also says `task` is informal and recommends narrower
terms such as `Problem`, `Training Example`, `Sample`, `Objective`, and `Loss`
when precision matters.

The implementation still uses:

```text
src/intrep/tasks/
```

with packages such as:

- `image_classification`
- `image_text_choice`
- `language_modeling`
- `grid_step_prediction`
- `shogi_policy_value`

## Why It Matters

Leaving the name as `tasks/` may be fine if everyone understands it as a
historical package name. The risk is that future code treats `tasks/` as the
place for anything related to a task, including source data, data selection,
training data bundles, checkpoints, or run outputs.

That would conflict with the current boundary work.

## Current Policy

Do not rename `tasks/` immediately.

Treat `tasks/` as a historical package name for problem-oriented model surfaces
until a concrete rename makes the code clearer.

## Acceptance Criteria

This issue can close when one of the following is true:

- the project explicitly accepts `tasks/` as a historical package name and the
  docs make that clear enough, or
- `tasks/` is renamed to a better package name without compatibility aliases.

## Non-Goals

- do not rename packages as part of this issue
- do not introduce a generic task framework
- do not move source data, generated data, checkpoints, or run artifacts into
  `tasks/`
