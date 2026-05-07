# Data Directory Boundary

Status: open.

## Issue

The top-level `data/` directory mixes dataset/source names with directories that
look like run or runtime-management artifacts.

Current examples:

- `data/runs/`
- `data/shogi/`
- `data/external/`

`data/runs/` is especially suspicious because `runs/` is now explicitly
disposable experiment output. If these files are datasets, the name is unclear.
If they are run outputs, they should not live under `data/`.

`data/shogi/` may be valid for shogi source records and experience stores, but
it also contains player/runtime configuration such as `player-registry.json`.
That may exceed the responsibility of a data directory.

`data/external/` is also suspicious because it groups unrelated corpora by
origin rather than by source/corpus name. It currently contains Tiny
Shakespeare, Gutenberg, FineWeb-Edu, TinyStories, WikiText-2, and probe or
failed-download files.

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
- Inspect what is currently under `data/external/`.
- Decide which entries are source data, derived datasets, experience stores,
  training views, runtime/player configuration, or disposable run output.
- Rename, move, or delete only after the responsibility is clear.

## Progress

2026-05-07:

- Qhapaq raw data was organized under `data/qhapaq/raw/kiffiles/` and
  `data/qhapaq/raw/results/`.
- The broken `Rota_orqha1018_2739Games.7z` 403 HTML file and duplicate raw-root
  CSV files were removed locally.
- `data/qhapaq/extracted/` was removed because it only held an unreferenced
  sample extraction.
- `data/qhapaq/processed/` was reduced to source-derived records:
  `qhapaq_all_games.jsonl` and `qhapaq_all_games_failures.jsonl`.
- `docs/datasets.md` now records that Qhapaq raw data is partial and that
  train/eval splits belong in Dataset Definitions or Training Views, not
  `processed/`.

Next focus:

- handle `data/external/` one source at a time
- remove or move obvious probe/failed-download files
- decide whether external corpora should become top-level source directories
  such as `data/tiny-shakespeare/`, `data/tinystories/`, and
  `data/wikitext-2/`

## Non-Goals

- Do not reorganize all local datasets at once.
- Do not introduce a broad artifact management system here.
- Do not move large local artifacts into git.

## Acceptance Criteria

This issue can close when the project has a clear rule for top-level `data/`
contents and the currently suspicious entries are either justified, renamed, or
moved out of `data/`.
