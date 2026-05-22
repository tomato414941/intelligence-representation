# Artifact Layout

This document defines where local artifacts belong. Dataset descriptions belong
in [datasets.md](datasets.md).

## Rules

- `data/<source>/raw/` holds externally acquired source files.
- `data/<source>/processed/` holds reusable source-derived records and failure
  logs.
- `data/shogi/records/<name>/` may hold durable normalized shogi game-record
  JSONL when the records are reused across more than one training bundle or
  evaluation workflow.
- `data/shogi/training-data-bundles/<name>/` holds durable shogi Training Data Bundles / Dataset
  Snapshots.
- `data/shogi/training-data-bundles/<name>/cache/` may hold rebuildable tensor
  caches derived from that bundle's `data-selection.json`.
- Keep `data/shogi/training-data-bundles/current/` as the normal active Training Data Bundle. Use
  `runs/` for temporary bundles, and add another durable bundle only when it has a
  concrete reuse reason.
- `runs/` holds run-specific inputs, outputs, metrics, and temporary
  checkpoints.
- Evaluation metrics and match outputs belong under `runs/` unless explicitly
  promoted.
- `models/<model-name>/` holds a long-lived loadable model entry artifact. See
  [model-artifacts.md](model-artifacts.md) for the type/checkpoint split.
- A model entry under `models/` must contain `manifest.json` and
  `components/` files for the model's input, core, and output modules.
- Do not put metrics, run logs, player presets, or lineage registries under
  `models/`.
- `tokenizers/<tokenizer-name>/tokenizer.json` holds a long-lived loadable text
  tokenizer when the tokenizer is reused outside the run that created it.
- A text checkpoint may embed the tokenizer payload instead of depending on a
  separate tokenizer artifact.
- If a checkpoint or run depends on a separate tokenizer, its metadata must make
  that tokenizer artifact traceable.
- Do not put training metrics, run logs, or broad tokenizer registries under
  `tokenizers/`.
- Do not use `data/external/`; use source-specific top-level directories.
- Add helper directories such as `images/` or `cache/` only when they solve an
  active problem for that source.

## Notes

`processed/` data may be regenerable, but it is not a runtime speed cache. It
is worth storing when it is a stable training or evaluation input, expensive
enough to rebuild, or needed to explain skipped source records.

`cache/` is not a source of truth. It should be rebuildable from `raw/`,
`processed/`, or a documented data selection / training data bundle.

Shogi game records store recorded game facts. Replay-derived traces,
position/legal-move expansions, and tensorized policy/value samples are caches
or problem artifacts, not source game records.

## Saved File Formats

File and artifact directory names identify the artifact's role, not its format
version. Do not put a schema version in names such as `checkpoint/`,
`tokenizer.json`, `manifest.json`, `metrics.json`, or cache labels such as
`legal-move`.

Reusable or loadable saved files store their format identifier inside the
payload as `schema_version`. Loaders should check `schema_version` before
trusting the payload. Run-local metrics and summaries may also use
`schema_version` when they are machine-read; human-only logs do not need a
schema marker.
