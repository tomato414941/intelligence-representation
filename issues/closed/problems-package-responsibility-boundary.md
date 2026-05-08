# Problem Package Responsibility Boundary

Status: closed.

## Issue

The old `Task Package` definition was too broad.

`docs/glossary.md` said a task package could bind model input construction,
output heads, targets, losses, metrics, training, checkpointing, and evaluation
for a task family. This kept packages practical, but it could also make
`tasks/` look like a place for everything related to a task.

## Why It Matters

Problem packages should not accidentally absorb responsibilities that belong to
data selection, target availability, training configuration, artifact storage,
or model/checkpoint management.

The risky parts of the current wording are especially:

- targets
- training
- checkpointing

Those can blur with source data, stored targets, run outputs, and model
artifacts.

## Direction

Consider narrowing `Problem Package` toward:

> A problem package is an objective-bound model surface for a problem family. It may
> own problem-specific model input construction, output heads, losses, metrics,
> and evaluation code when those boundaries are not reusable elsewhere.

This would keep problem-specific code practical while avoiding "problem owns
everything" drift.

## Acceptance Criteria

- [x] Decide whether the glossary definition should be narrowed.
- [x] Decide whether `targets`, `training`, and `checkpointing` should remain in
  the definition, be qualified, or be removed.
- [x] Keep the package separate from dataset instances, run outputs, and generated
  artifacts.

## Resolution

The glossary now treats `Problem Package` as the package name for a
problem-oriented model surface. It no longer claims ownership of targets,
training, checkpointing, artifact storage, or data selection.

## Non-Goals

- introduce a broad task framework
- move shogi arena/runtime code into this repository
