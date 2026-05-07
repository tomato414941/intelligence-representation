# Data Shogi Directory

Status: open.

## Issue

`data/shogi/` contains several shogi-specific data areas, including generated
experience, training views, copied records, and possibly runtime/player
configuration.

Some of this belongs under `data/`, but the directory should not become a
catch-all for shogi runtime configuration or historical run artifacts.

## Why It Matters

The project now treats `data/` as the home for source data, processed data,
durable experience, training views, and useful derived data. Runtime
configuration and disposable run output should live elsewhere.

If `data/shogi/` stays broad, future shogi work may put player picks, checkpoint
references, run summaries, generated records, training views, and caches in the
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
  tracks the player registry specifically.
- [`experience-store-training-view-boundary.md`](experience-store-training-view-boundary.md)
  tracks whether the store/view lifecycle should generalize beyond shogi.
- [`shogi-training-view-tensor-cache.md`](shogi-training-view-tensor-cache.md)
  tracks shogi training cache performance.

## Acceptance Criteria

This issue can close when direct children under `data/shogi/` have clear data
responsibilities and runtime/player configuration is either removed, moved, or
tracked by a narrower open issue.

## Non-Goals

- redesign shogi Experience Store
- solve player registry policy
- introduce a generic Experience Store or Training View abstraction
- move Qhapaq data into `data/shogi/`
