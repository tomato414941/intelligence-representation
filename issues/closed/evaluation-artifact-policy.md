# Evaluation Artifact Policy

Status: closed.

## Issue

Evaluation outputs can be produced by task metrics, shogi arena matches, and
other checks, but it is not yet clear which outputs are disposable run output
and which should be kept with long-lived models or datasets.

## Why It Matters

Model comparison depends on evaluation results. If useful evaluation outputs
stay only under disposable `runs/`, they may be lost. If every evaluation output
is promoted, artifact management becomes noisy.

## Scope

- Decide where task metric JSON files belong.
- Decide where shogi match or arena evaluation outputs belong.
- Decide which evaluation summaries, if any, should be copied next to a
  long-lived model checkpoint.
- Keep the rule small enough to use during local experiments.

## Acceptance Criteria

This issue can close when evaluation outputs have clear default and long-lived
locations.

## Resolution

Evaluation outputs are run artifacts by default.

- Task metrics and match outputs belong under `runs/`.
- `models/` stores loadable checkpoints only.
- If an evaluation output becomes a long-lived input for future work, promote it
  through a separate explicit decision.

The default placement rule is recorded in
[`../../docs/artifact-layout.md`](../../docs/artifact-layout.md).
