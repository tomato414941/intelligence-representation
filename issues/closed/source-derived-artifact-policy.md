# Source-Derived Artifact Policy

Status: closed.

## Issue

The project keeps some source-derived artifacts, such as processed records and
failure logs, but the boundary between source data, processed data, and cache is
still easy to blur.

## Why It Matters

Source-derived records can be smaller and easier to audit than task-specific
caches. Failure logs can explain skipped inputs. If these artifacts are treated
like disposable run output, future dataset rebuilding becomes harder. If they
are treated like raw source data, the source of truth becomes unclear.

## Scope

- Decide where source-derived records belong.
- Decide where failure logs belong.
- Clarify how source-derived records differ from caches.
- Ensure Qhapaq processed data follows the chosen policy.

## Non-Goals

- Do not redesign `data/` layout broadly.
- Do not add a generic artifact store.
- Do not solve shogi Training View tensor caching.

## Acceptance Criteria

This issue can close when source-derived records and failure logs have clear
locations and lifecycle rules.

## Resolution

Source-derived records and failure logs belong under
`data/<source>/processed/`.

They may be regenerable, but they are not runtime speed caches. They are kept
when they are stable training or evaluation inputs, expensive enough to rebuild,
or needed to explain skipped source records.

The default placement rule is recorded in
[`../../docs/artifact-layout.md`](../../docs/artifact-layout.md).

Current Qhapaq processed artifacts follow this policy:

- `data/qhapaq/processed/qhapaq_all_games.jsonl`
- `data/qhapaq/processed/qhapaq_all_games_failures.jsonl`
