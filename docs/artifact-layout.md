# Artifact Layout

This document defines where local artifacts belong. Dataset descriptions belong
in [datasets.md](datasets.md).

## Rules

- `data/<source>/raw/` holds externally acquired source files.
- `data/<source>/processed/` holds reusable derived records or examples.
- `runs/` holds run-specific inputs, outputs, metrics, and temporary
  checkpoints.
- Evaluation metrics and match outputs belong under `runs/` unless explicitly
  promoted.
- `models/<model-name>/checkpoint.pt` holds a long-lived loadable checkpoint.
- `checkpoint.pt` under `models/` must contain the schema, model config, and
  state dict needed to load it.
- Do not put metrics, run logs, player presets, or lineage registries under
  `models/`.
- Do not use `data/external/`; use source-specific top-level directories.
- Add helper directories such as `images/` or `cache/` only when they solve an
  active problem for that source.

## Notes

`processed/` data may be regenerable. It is still worth storing when it is a
stable training or evaluation input, expensive enough to rebuild, or needed for
fair comparisons.

`cache/` is not a source of truth. It should be rebuildable from `raw/`,
`processed/`, or a documented dataset/training view.
