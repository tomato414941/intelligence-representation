# Shogi Training Data Bundle Train/Eval Overlap

Status: open. Priority: low.

## Issue

The current shogi Training Data Bundle can contain positions in eval that also appear
in train.

Measured on `data/shogi/training-data-bundles/current/`:

- train/eval position overlap count: 747
- train/eval position overlap ratio: 0.1506

## Why It Matters

This is evaluation leakage, not Experience Store forgetting. If eval contains
positions already present in train, evaluation can overstate generalization.

## Initial Policy

Keep this low priority for now. The current eval is useful for local iteration,
and no high-stakes model comparison depends on it yet.

When needed, solve this in Training Data Bundle creation, not by deleting Experience
Store records.

## Acceptance Criteria

This issue can close when shogi Training Data Bundle creation can produce an eval set
that excludes positions already present in train.
