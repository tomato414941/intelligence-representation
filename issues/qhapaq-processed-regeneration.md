# Qhapaq Processed Regeneration

Status: open. Priority: low.

## Issue

Qhapaq raw data can now contain multiple `.7z` kif archives under
`data/qhapaq/raw/kiffiles/`, but there is no clear procedure for regenerating
`data/qhapaq/processed/qhapaq_all_games.jsonl` from the current raw archive
set.

The existing KIF conversion code can convert KIF files to `ShogiGameRecord`
JSONL, but it does not own:

- extracting selected Qhapaq `.7z` archives
- collecting KIF paths from temporary extraction directories
- preserving or reporting conversion failures
- writing a refreshed `qhapaq_all_games.jsonl`
- documenting whether processed output covers all local raw archives or only a
  selected subset

## Why It Matters

`processed/` now intentionally contains only source-derived records:

```text
data/qhapaq/processed/qhapaq_all_games.jsonl
data/qhapaq/processed/qhapaq_all_games_failures.jsonl
```

After adding new raw `.7z` archives, it is unclear whether these processed
records are stale or complete relative to local raw data.

Without a clear regeneration path, Qhapaq raw cleanup can make the directory
look organized while the processed records still reflect an older subset.

## Initial Policy

This is intentionally deferred for now. The current processed data is usable,
and Qhapaq is not the main near-term bottleneck.

Do not keep persistent extracted/interim KIF directories unless conversion cost
or debugging requires it.

Prefer a simple regeneration command or documented procedure that extracts raw
archives into a temporary directory, converts KIF files to source-derived
records, records failures, and writes the processed JSONL outputs.

## Acceptance Criteria

This issue can close when the project has a simple way to regenerate
`qhapaq_all_games.jsonl` from the intended local raw Qhapaq archive set, and the
processed output records which raw archives it covers.

## Non-Goals

- download every Qhapaq archive
- introduce a generic dataset pipeline framework
- create persistent `interim/` output by default
- create train/eval splits inside `processed/`
