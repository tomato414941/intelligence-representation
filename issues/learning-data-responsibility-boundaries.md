# Learning Data Responsibility Boundaries

Status: open.

## Issue

The project needs clearer responsibilities between source data and training.
Naming should follow responsibility, not the other way around.

This matters because names such as `Dataset Definition`, `Training View`, and
`Example` can become too broad if they are assigned before the responsibilities
are separated.

## Responsibility Draft

Separate these responsibilities before promoting names into glossary terms:

- source storage: preserve source-side records without reshaping them around
  one objective, model, or run
- target availability: identify or store values that can be used as targets
- data inclusion: decide which source records, training examples, or stored
  targets are included for training or evaluation
- example construction: shape included source records and targets into
  objective-specific training examples
- runtime sampling: adapt training examples into PyTorch samples
- batching: group samples for training or evaluation
- optimization: turn objectives into losses or learning signals
- artifact storage: store run outputs, caches, checkpoints, and metrics

## Current Concern

`Dataset Definition` may be too broad as a name if it absorbs data inclusion,
example construction, target selection, sampling, or training configuration.

The first boundary to decide is responsibility, not the final name.

## Acceptance Criteria

- Decide which responsibilities need project-level names.
- Decide whether `Dataset Definition` should remain, be narrowed, or be
  replaced by a more precise term.
- Promote only stable terms into `docs/glossary.md`.
- Keep relationship explanations in `docs/learning-boundaries.md` only after
  the responsibilities are stable enough.

## Non-Goals

- introduce a generic dataset framework
- rename existing classes immediately
- redesign shogi training data storage in this issue
