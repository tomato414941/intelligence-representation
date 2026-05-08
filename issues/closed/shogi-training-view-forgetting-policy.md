# Shogi Training View Forgetting Policy

Status: closed.

## Issue

Shogi Experience Store should keep raw generated experience, including duplicate
positions and old weak-model games. Deleting from the store is a heavy operation
and makes later review harder.

Training View, however, needs a way to exclude data for a specific
training run. Current data already shows this pressure:

- repeated deterministic openings create many duplicate positions
- old weak-model experience may become undesirable for a later view

## Why It Matters

The store is a historical source of experience. A Training View is the dataset a
run actually trains and evaluates on.

If exclusion happens by deleting store records, the project loses auditability.
If exclusion is impossible in Training View creation, models may keep training
on stale or overrepresented examples.

## Initial Policy

Exclude from the Training View, not from the Experience Store.

Keep the first implementation narrow:

- optionally cap or exclude repeated positions
- optionally exclude known weak or obsolete actor-pair slices

Do not add broad retention policies, weighted sampling, or generic data
lifecycle machinery until a concrete run needs them.

## Acceptance Criteria

This issue can close when a concrete training run needs exclusion and shogi
Training View creation can express that exclusion without deleting Experience
Store records.

## Related

Train/eval overlap is evaluation leakage rather than Training View forgetting.
It is tracked separately in
[`shogi-training-view-train-eval-overlap.md`](shogi-training-view-train-eval-overlap.md).

## Resolution

Closed by splitting the concerns into narrower issues:

- train/eval leakage is tracked by
  [`../shogi-training-view-train-eval-overlap.md`](../shogi-training-view-train-eval-overlap.md)
  and is low priority for now.
- actor-pair inclusion, exclusion, and caps belong to
  [`../shogi-training-view-source-mix.md`](../shogi-training-view-source-mix.md).

The remaining principle is simple: do not delete Experience Store records just
to shape a specific Training View.
