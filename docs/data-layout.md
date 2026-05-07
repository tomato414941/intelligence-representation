# Data Layout

This document defines where local data artifacts belong. Dataset descriptions
belong in [datasets.md](datasets.md).

## Rules

- `data/<source>/raw/` holds externally acquired source files.
- `data/<source>/processed/` holds reusable derived records or examples.
- `runs/` holds run-specific inputs, outputs, metrics, and checkpoints.
- Do not use `data/external/`; use source-specific top-level directories.
- Add helper directories such as `images/` or `cache/` only when they solve an
  active problem for that source.

## Notes

`processed/` data may be regenerable. It is still worth storing when it is a
stable training or evaluation input, expensive enough to rebuild, or needed for
fair comparisons.

`cache/` is not a source of truth. It should be rebuildable from `raw/`,
`processed/`, or a documented dataset/training view.
