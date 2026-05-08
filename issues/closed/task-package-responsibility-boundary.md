# Task Package Responsibility Boundary

Status: closed.

## Issue

The current `Task Package` definition may be too broad.

`docs/glossary.md` says a task package can bind model input construction,
output heads, targets, losses, metrics, training, checkpointing, and evaluation
for a task family. This keeps task packages practical, but it can also make
`tasks/` look like a place for everything related to a task.

## Why It Matters

Task packages should not accidentally absorb responsibilities that belong to
data selection, target availability, training configuration, artifact storage,
or model/checkpoint management.

The risky parts of the current wording are especially:

- targets
- training
- checkpointing

Those can blur with source data, stored targets, run outputs, and model
artifacts.

## Direction

Consider narrowing `Task Package` toward:

> A task package is an objective-bound model surface for a task family. It may
> own task-specific model input construction, output heads, losses, metrics,
> and evaluation code when those boundaries are not reusable elsewhere.

This would keep task-specific code practical while avoiding "task owns
everything" drift.

## Acceptance Criteria

- [x] Decide whether the glossary definition should be narrowed.
- [x] Decide whether `targets`, `training`, and `checkpointing` should remain in
  the definition, be qualified, or be removed.
- [x] Keep `tasks/` separate from dataset instances, run outputs, and generated
  artifacts.

## Resolution

The glossary now treats `Task Package` as a historical package name for a
problem-oriented model surface, not as a precise project boundary. It no longer
claims ownership of targets, training, checkpointing, artifact storage, or data
selection. `docs/architecture.md` now says `task` is informal and points readers
to narrower terms when precision matters.

## Non-Goals

- rename existing task packages
- introduce a broad task framework
- move shogi arena/runtime code into this repository
