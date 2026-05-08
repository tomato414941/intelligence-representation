# Problems Package Name Boundary

Status: closed.

## Issue

`tasks/` remained as a real package name, but the project treated `task` as an
informal word rather than a precise boundary.

This is different from the closed `problems-package-responsibility-boundary.md`
issue. That issue narrowed what a package may own. This issue was only about
whether the package name `tasks/` still fit.

## Current State

`docs/glossary.md` said `Task Package` was a historical package name for a
problem-oriented model surface.

`docs/architecture.md` also says `task` is informal and recommends narrower
terms such as `Problem`, `Training Example`, `Sample`, `Objective`, and `Loss`
when precision matters.

The implementation used:

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

## Resolution

Renamed `src/intrep/tasks/` to `src/intrep/problems/` without compatibility
aliases.

`Problem Package` is now the package-level term. `task` remains an informal
word in prose, but it is no longer the package boundary.

## Acceptance Criteria

This issue can close when one of the following is true:

- [x] `tasks/` is renamed to a better package name without compatibility aliases.

## Non-Goals

- do not introduce a generic task framework
- do not move source data, generated data, checkpoints, or run artifacts into
  `problems/`
