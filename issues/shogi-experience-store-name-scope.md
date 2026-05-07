# Shogi Experience Store Name Scope

Status: open.

## Issue

Current shogi Experience Store names do not clearly separate store role from
source details.

Current stores:

```text
data/shogi/experiences/main/
data/shogi/experiences/heldout-yaneuraou-self/
```

`main` is vague. `heldout-yaneuraou-self` is more descriptive, but it mixes a
usage role (`heldout`) with source detail (`yaneuraou-self`). It is unclear
whether the store name should describe:

- the store's operational role
- the actor/source mix
- a train/eval purpose
- a temporary experiment-specific split

## Why It Matters

The current split was likely chosen because it is easy to keep training and
evaluation source records separate at store creation time. That is acceptable
for the current KISS workflow.

However, unclear store names make it harder to tell whether a directory is:

- a durable source of generated experience
- a training candidate store
- an evaluation candidate store
- a source-specific collection

Actor mix and source details can be recorded in `manifest.json`; they do not
necessarily need to be encoded in the directory name.

## Initial Policy

Do not add a registry or a complex naming scheme.

Prefer short role-oriented names if the project continues to split stores at
the Experience Store level, for example:

```text
data/shogi/experiences/train/
data/shogi/experiences/eval/
```

Keep source details such as `yaneuraou:self` in manifest metadata, not in the
store directory name, unless a concrete workflow needs source-specific store
names.

## Acceptance Criteria

This issue can close when the project decides a simple Experience Store naming
policy and the current store names either follow it or are explicitly kept as an
exception.

The decision should state whether train/eval-like separation belongs at the
Experience Store level for now, or only at Training View creation time.

## Non-Goals

- redesign Experience Store
- merge train/eval source stores
- introduce a store registry
- solve Dataset / Training View naming
