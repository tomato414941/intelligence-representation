# Artifact Layout

This document defines where local artifacts belong. Dataset descriptions belong
in [datasets.md](datasets.md).

## Rules

- `data/<source>/raw/` holds externally acquired source files.
- `data/<source>/processed/` holds reusable source-derived records and failure
  logs.
- `data/shogi/datasets/<name>/` holds durable shogi Training Views / Dataset
  Snapshots.
- Keep `data/shogi/datasets/current/` as the normal active Training View. Use
  `runs/` for temporary views, and add another durable view only when it has a
  concrete reuse reason.
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

`processed/` data may be regenerable, but it is not a runtime speed cache. It
is worth storing when it is a stable training or evaluation input, expensive
enough to rebuild, or needed to explain skipped source records.

`cache/` is not a source of truth. It should be rebuildable from `raw/`,
`processed/`, or a documented data selection / training view.
