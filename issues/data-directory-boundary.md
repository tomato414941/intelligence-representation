# Data Directory Boundary

Status: open.

## Issue

The top-level `data/` directory mixes dataset/source names with directories that
look like run or runtime-management artifacts.

Current examples:

- `data/runs/`
- `data/shogi/`

`data/runs/` is especially suspicious because `runs/` is now explicitly
disposable experiment output. If these files are datasets, the name is unclear.
If they are run outputs, they should not live under `data/`.

`data/shogi/` may be valid for shogi source records and experience stores, but
it also contains player/runtime configuration such as `player-registry.json`.
That may exceed the responsibility of a data directory.

## Why It Matters

`data/` should hold source data, generated datasets, experience stores, and
training views. It should not become a second run-output tree or a catch-all
for evaluation/runtime configuration.

If this boundary remains unclear, future generated data, model checkpoints,
player registries, and run summaries may drift into whichever directory happens
to exist.

## Scope

- Inspect what is currently under `data/runs/`.
- Inspect what is currently under `data/shogi/`.
- Decide which entries are source data, derived datasets, experience stores,
  training views, runtime/player configuration, or disposable run output.
- Rename, move, or delete only after the responsibility is clear.

## Non-Goals

- Do not reorganize all local datasets at once.
- Do not introduce a broad artifact management system here.
- Do not move large local artifacts into git.

## Acceptance Criteria

This issue can close when the project has a clear rule for top-level `data/`
contents and the currently suspicious entries are either justified, renamed, or
moved out of `data/`.
