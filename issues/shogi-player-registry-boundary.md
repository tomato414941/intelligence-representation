# Shogi Player Registry Boundary

Status: open.

## Issue

`data/shogi/player-registry.json` currently acts like a manual player pick list,
but its responsibility and location are unclear.

It contains:

- checkpoint players
- USI engine players
- notes and lineage-like fields such as `parent`, `dataset`, and
  `best_eval_step`

The file currently validates, but it is not used by the main training, battle,
or RL-cycle paths. It is closer to optional evaluation/runtime configuration
than source data.

## Why It Matters

The current registry conflicts with newer project boundaries:

- `data/` should not become a catch-all for runtime configuration.
- `runs/` is disposable, but checkpoint players in the registry still point to
  `runs/shogi/...`.
- The registry is not a model registry; it should not become a second source of
  truth for model lineage or dataset history.
- Long-lived checkpoints can live under `models/`, but player selection is a
  separate concern.

## Scope

- Decide whether a player registry is needed at all.
- If kept, define it as a small evaluation/runtime pick list, not model
  management.
- Decide where the registry belongs if not under `data/shogi/`.
- Remove or avoid lineage-like fields unless a concrete workflow needs them.
- Ensure checkpoint players do not depend on disposable `runs/` paths.

## Non-Goals

- Do not build a full model registry.
- Do not add parent-run or training-history tracking.
- Do not solve all `data/` directory layout issues here; the prior cleanup is
  summarized in `data-layout-cleanup-summary.md`.

## Acceptance Criteria

This issue can close when the project either removes the player registry or
keeps a clearly scoped player pick list whose location and checkpoint references
match the `runs/` and `models/` artifact policies.
