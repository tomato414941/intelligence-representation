# Shogi Training View Forgetting Policy

Status: open.

## Issue

Shogi Experience Store should keep raw generated experience, including duplicate
positions and old weak-model games. Deleting from the store is a heavy operation
and makes later review harder.

Training View, however, needs a way to forget or exclude data for a specific
training/evaluation run. Current data already shows this pressure:

- repeated deterministic openings create many duplicate positions
- eval contains positions that also appear in train
- old weak-model experience may become undesirable for a later view

## Why It Matters

The store is a historical source of experience. A Training View is the dataset a
run actually trains and evaluates on.

If forgetting happens by deleting store records, the project loses auditability.
If forgetting is impossible in Training View creation, models may keep training
on stale, duplicated, or leaked examples.

## Initial Policy

Forget from the Training View, not from the Experience Store.

Keep the first implementation narrow:

- exclude eval positions that also appear in train
- optionally exclude duplicate eval positions

Do not add broad retention policies, weighted sampling, or generic data
lifecycle machinery until a concrete run needs them.

## Acceptance Criteria

This issue can close when shogi Training View creation can prevent train/eval
position overlap without deleting Experience Store records.

