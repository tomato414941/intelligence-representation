# Model Artifact Policy

Status: closed.

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

## Resolution

The minimal model artifact policy is now documented in
[`../../docs/artifact-layout.md`](../../docs/artifact-layout.md):

- `models/<model-name>/checkpoint.pt` is the long-lived model artifact.
- The checkpoint must contain the schema, model config, and state dict needed to
  load it.
- Metrics, run logs, player presets, and lineage registries do not belong under
  `models/`.

The current checkpoint at `models/d32-h64-heads4-l1/checkpoint.pt` follows this
policy and was verified to load with the current shogi policy-value checkpoint
loader.
