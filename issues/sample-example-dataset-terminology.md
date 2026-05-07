# Sample / Example / Dataset Terminology

Status: open.

## Issue

The project uses terms such as `Example`, `Dataset`, `Experience`, `Record`,
`Training View`, and `Dataset Definition`, but the boundaries are still easy to
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
- How should glossary entries align with PyTorch `Dataset` / `DataLoader`
  language?
- Which existing class names should stay as-is because they are already
  specific and useful?

## Acceptance Criteria

This issue can close when the glossary and data layout docs clearly distinguish:

- source record
- experience
- sample
- supervised training example
- PyTorch `Dataset`
- file-backed dataset or training view
- cache

## Non-Goals

- rename all existing `Example` classes immediately
- redesign shogi Experience Store
- introduce a generic dataset framework
