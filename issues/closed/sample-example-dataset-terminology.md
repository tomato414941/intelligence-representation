# Sample / Example / Dataset Terminology

Status: closed.

## Issue

The project uses terms such as `Example`, `Dataset`, `Experience`, `Record`,
`Training Data Bundle`, and `Data Selection`, but the boundaries are still easy to
blur.

The current concern is that `example` may be too narrow as the general name for
one training item. In PyTorch, the closest practical concept is the item
returned by `Dataset.__getitem__`, often called a sample. `example` is still
reasonable for supervised records that already carry an input and target, such
as `ImageClassificationExample` or `ShogiMoveChoiceExample`.

RL experience and self-play records are broader: they may contain state,
action, reward, next state, search output, actor metadata, and raw observations
before any supervised target is selected.

## Why It Matters

If these names stay vague, storage and code boundaries can drift:

- raw records may be treated as training examples
- examples may be treated as durable experience
- file-backed dataset snapshots may be confused with PyTorch `Dataset`
  objects
- cache artifacts may look like source data

## Questions

- Should the general one-item concept be `sample` instead of `example`?
- Should `example` be reserved for supervised training-ready records?
- Which existing class names should stay as-is because they are already
  specific and useful?

## Progress

- The glossary now defines PyTorch `Dataset` as an indexed-sample adapter, not
  the source of truth for raw data, target generation, split policy, or
  learning intent.
- `Data Selection` is now the glossary term for the data inclusion boundary.
- `learning-boundaries.md` now states that PyTorch `Dataset` objects adapt an
  already-selected training or evaluation set into indexed samples.
- The glossary now distinguishes `Training Example` as the meaning-level
  objective-shaped unit from `Sample` as the runtime item returned by
  `Dataset.__getitem__`.

## Resolution

The terminology boundary is now stable enough:

- `Source Record` and `Experience` are defined separately.
- `Data Selection` is the inclusion boundary.
- `Training Example` is the objective-specific input/target-or-feedback unit.
- `Sample` is the PyTorch runtime item returned by `Dataset.__getitem__`.
- PyTorch `Dataset` is an indexed-sample adapter.
- `cache/` remains a rebuildable artifact, not a source of truth.

Remaining code-level responsibility mixing is tracked separately in
`training-example-responsibility-mixing.md`.

## Acceptance Criteria

This issue can close when the glossary and data layout docs clearly distinguish:

- source record
- experience
- sample
- supervised training example
- PyTorch `Dataset`
- file-backed dataset or training data bundle
- cache

## Non-Goals

- rename all existing `Example` classes immediately
- redesign shogi Experience Store
- introduce a generic dataset framework
