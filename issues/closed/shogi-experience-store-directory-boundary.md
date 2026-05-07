# Shogi Experience Store Directory Boundary

Status: closed.

## Issue

`data/shogi/experiences/` should contain only shogi Experience Store
directories, but it currently also contains at least one directory that does not
match the store shape.

Expected store shape:

```text
data/shogi/experiences/<store-name>/
  games.jsonl
  history.jsonl
  manifest.json
```

Current mismatch:

```text
data/shogi/experiences/mixed-g1000-eval-yaneuraou-g100/
```

This directory contains source JSONL files for a mixed training/eval setup, but
it does not have the canonical Experience Store files above.

## Why It Matters

The project uses Experience Store to mean durable generated experience. If
non-store data lives under `experiences/`, it becomes unclear which directories
are canonical stores and which are temporary inputs, source mixes, or training
view material.

That weakens the boundary between:

- Experience Store: persistent raw experience
- Training View: fixed train/eval snapshot
- run output or ad hoc source files: disposable or intermediate material

## Initial Policy

Keep the name `Experience Store` for now.

Fix the directory boundary first: every direct child of
`data/shogi/experiences/` should either be a valid Experience Store or be moved
elsewhere.

Do not introduce a broader storage abstraction for this issue.

## Acceptance Criteria

This issue can close when:

- every direct child under `data/shogi/experiences/` follows the store shape, or
  the project explicitly documents an exception
- non-store mixed source files are moved to a more appropriate location or
  removed if no longer needed
- scripts and docs no longer imply that non-store directories under
  `experiences/` are valid Experience Stores

## Resolution

Closed on 2026-05-07.

Removed the local ignored non-store directory:

```text
data/shogi/experiences/mixed-g1000-eval-yaneuraou-g100/
```

It was not tracked by git and was not referenced outside this issue. The
remaining direct children under `data/shogi/experiences/` follow the expected
Experience Store shape.

## Non-Goals

- rename Experience Store
- implement Replay Buffer or sampling policy
- change ShogiGameRecord schema
- redesign Training View
