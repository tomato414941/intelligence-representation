# Data Runs Directory

Status: open.

## Issue

`data/runs/` exists under the data tree, but `runs/` is supposed to be
disposable experiment output.

If files under `data/runs/` are run outputs, they should not live under
`data/`. If they are reusable data artifacts, their source and responsibility
should be explicit and they should move to a source-specific location.

## Why It Matters

Keeping a `runs` directory under `data/` blurs the project boundary:

- `data/` should hold source data, processed data, durable experience, training
  views, or explicitly useful derived data
- `runs/` should hold run-specific inputs, outputs, metrics, checkpoints, and
  scratch artifacts

If `data/runs/` remains, future work may keep adding experiment output under
the data tree.

## Scope

- Inspect the current contents of `data/runs/`.
- Classify each file as disposable run output, reusable processed data, or
  obsolete artifact.
- Delete disposable or obsolete files.
- Move reusable data artifacts to source-specific paths if they still matter.
- Remove `data/runs/` when it is empty.

## Acceptance Criteria

This issue can close when `data/runs/` is gone or has been renamed to a
source-specific data location with clear responsibility.

## Non-Goals

- redesign the global `runs/` directory
- create a general artifact store
- solve checkpoint/model registry policy
