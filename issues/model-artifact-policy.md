# Model Artifact Policy

Status: open.

## Issue

The project now has a `models/` directory for long-lived checkpoints, but it
does not yet define what must live beside a checkpoint.

## Why It Matters

A checkpoint is useful only if future code can load it and a reader can
understand what it represents. Saving only `checkpoint.pt` may be enough when
the checkpoint contains config and schema, but metrics, evaluation context, and
source data scope can still be unclear.

## Scope

- Decide the minimal contents of a long-lived model directory.
- Decide whether `checkpoint.pt` must contain all required model config.
- Decide whether model-level metadata should include training metrics,
  evaluation summaries, git commit, or dataset/view references.
- Define when a run checkpoint should be promoted from `runs/` to `models/`.

## Non-Goals

- Do not build a full model registry.
- Do not track player presets.
- Do not require storing every run checkpoint permanently.

## Acceptance Criteria

This issue can close when `models/` has a documented minimal structure and the
current long-lived checkpoint follows it.
