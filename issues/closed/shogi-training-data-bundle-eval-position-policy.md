# Shogi Training Data Bundle Eval Position Policy

Status: closed. Priority: low.

## Issue

Shogi Training Data Bundle creation needs to make the eval position policy
explicit. Some eval sets intentionally allow positions that also appear in
train, while held-out-position eval sets should exclude train positions.

The earlier symptom was measured on `data/shogi/training-data-bundles/current/`
when that bundle was present:

- train/eval position overlap count: 747
- train/eval position overlap ratio: 0.1506

## Why It Matters

This can be evaluation leakage, not Experience Store forgetting. If eval
contains positions already present in train, evaluation can overstate
held-out-position generalization. But always removing overlap is also not
universally correct, because seen-position or distribution-consistency eval may
intentionally allow repeated openings or repeated evidence.

## Policy

Experience Store keeps raw source experience and does not delete overlapping
positions for evaluation convenience.

Training Data Bundle creation owns eval position policy because it creates the
fixed train/eval input artifact. The first supported policies are:

- `allow_overlap`: keep current behavior and record overlap stats.
- `exclude_train_position_games`: remove eval games that contain any position
  already present in the selected train games.

Exclusion is game-level, not transition-level, so `ShogiGameRecord` source
records are not sliced into partial games.

## Acceptance Criteria

This issue can close when shogi Training Data Bundle creation can express and
record eval position policy, and can produce an eval set that excludes games
containing train positions.

## Resolution

`create_shogi_training_data_bundle()` now accepts `eval_position_policy`, and
`scripts/create_shogi_training_data_bundle.py` exposes `--eval-position-policy`.
The manifest records `eval_position_policy`,
`selected_eval_games_before_position_policy`, and
`skipped_eval_games_for_train_position_overlap` in addition to the existing
train/eval position overlap stats.
