# Qhapaq Processed Regeneration

Status: closed.

## Issue

Qhapaq raw data can now contain multiple `.7z` kif archives under
`data/qhapaq/raw/kiffiles/`, but there was no clear procedure for regenerating
the processed source records from the current raw archive set.

The existing KIF conversion code could convert KIF files, but it did not own:

- extracting selected Qhapaq `.7z` archives
- collecting KIF paths from temporary extraction directories
- preserving or reporting conversion failures
- writing refreshed processed records
- documenting whether processed output covers all local raw archives or only a
  selected subset

## Why It Matters

`processed/` now intentionally contains only source-derived records:

```text
data/qhapaq/processed/qhapaq_games.jsonl
data/qhapaq/processed/qhapaq_game_failures.jsonl
data/qhapaq/processed/manifest.json
```

After adding new raw `.7z` archives, it is unclear whether these processed
records are stale or complete relative to local raw data.

Without a clear regeneration path, Qhapaq raw cleanup can make the directory
look organized while the processed records still reflect an older subset.

## Initial Policy

Do not keep persistent extracted/interim KIF directories unless conversion cost
or debugging requires it.

Prefer a simple regeneration command or documented procedure that extracts raw
archives into a temporary directory, converts KIF files to source-derived
records, records failures, and writes the processed JSONL outputs.

## Acceptance Criteria

This issue can close when the project has a simple way to regenerate processed
records from the intended local raw Qhapaq archive set, and the processed output
records which raw archives it covers.

## Resolution

`scripts/prepare_qhapaq_shogi_records.py` regenerates compact Qhapaq source
records from local `.7z` archives using temporary extraction directories.

The generated local outputs are:

```text
data/qhapaq/processed/qhapaq_games.jsonl
data/qhapaq/processed/qhapaq_game_failures.jsonl
data/qhapaq/processed/manifest.json
```

On 2026-05-15, the local refresh processed 64 archives into 39,740 games and
5,213,204 moves, with 0 conversion failures. The source page had 65 `.7z`
links; `Rota_orqha1018_2739Games.7z` returned HTTP 403 and is recorded as
unavailable in the local raw manifest.

## Non-Goals

- download every Qhapaq archive
- introduce a generic dataset pipeline framework
- create persistent `interim/` output by default
- create train/eval splits inside `processed/`
