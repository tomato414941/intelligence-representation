# Data Shogi Directory

Status: closed.

## Issue

`data/shogi/` contains several shogi-specific data areas, including generated
experience, training data bundles, copied records, and possibly runtime/player
configuration.

Some of this belongs under `data/`, but the directory should not become a
catch-all for shogi runtime configuration or historical run artifacts.

## Why It Matters

The project now treats `data/` as the home for source data, processed data,
durable experience, training data bundles, and useful derived data. Runtime
configuration and disposable run output should live elsewhere.

If `data/shogi/` stays broad, future shogi work may put player picks, checkpoint
references, run summaries, generated records, training data bundles, and caches in the
same place without clear responsibility.

## Scope

- Inspect current children under `data/shogi/`.
- Keep valid shogi data stores or training inputs under `data/shogi/`.
- Identify runtime/player configuration that should move elsewhere or be
  covered by a narrower issue.
- Identify copied records or historical artifacts that can be deleted or moved.
- Leave source-specific external data, such as Qhapaq, under its own source
  directory.

## Related Issues

- [`shogi-player-registry-boundary.md`](shogi-player-registry-boundary.md)
  later removed the player registry.
- [`../mixed-source-store-continual-learning.md`](../mixed-source-store-continual-learning.md)
  tracks whether a future mixed source store should exist beyond shogi.
- [`shogi-training-data-bundle-tensor-cache.md`](shogi-training-data-bundle-tensor-cache.md)
  tracks shogi training cache performance.

## Acceptance Criteria

This issue can close when direct children under `data/shogi/` have clear data
responsibilities and runtime/player configuration is either removed, moved, or
tracked by a narrower open issue.

## Resolution

The current direct children under `data/shogi/` were inspected:

- `training-data-bundles/`: durable Training Data Bundles.
- `experiences/`: shogi Experience Stores.
- `records/`: source-derived copied record sets.
- `player-registry.json`: removed by
  [`shogi-player-registry-boundary.md`](shogi-player-registry-boundary.md).

This closes the immediate directory-content audit. The remaining broader
concern that `data/shogi/` is a world-level bucket rather than a source-level
data directory is tracked separately in
[`../shogi-world-data-directory-boundary.md`](../shogi-world-data-directory-boundary.md).

## Non-Goals

- redesign shogi Experience Store
- reintroduce player registry policy
- introduce a generic Experience Store or Training Data Bundle abstraction
- move Qhapaq data into `data/shogi/`
