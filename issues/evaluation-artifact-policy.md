# Evaluation Artifact Policy

Status: open.

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

## Non-Goals

- Do not build a leaderboard.
- Do not reintroduce player registry management.
- Do not decide model promotion criteria beyond evaluation artifact handling.

## Acceptance Criteria

This issue can close when evaluation outputs have clear default and long-lived
locations.
