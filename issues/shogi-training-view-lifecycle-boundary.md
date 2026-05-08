# Shogi Training View Lifecycle Boundary

Status: open.

## Issue

The project should clearly distinguish durable Training Views from run-local
datasets.

A view under `data/shogi/datasets/<name>/` is not temporary; it is a fixed,
reusable snapshot. A dataset written under `runs/.../` is run-local and
disposable.

Calling both "temporary views" makes the storage boundary unclear.

## Why It Matters

`runs/` should be disposable experiment output. If run-local datasets and
durable Training Views are described with the same name, it becomes unclear
which files can be deleted and which files are intended to be reused.

The distinction should be simple:

- durable fixed view: `data/shogi/datasets/<name>/`
- run-local dataset: `runs/.../dataset.json` and adjacent source files

## Initial Policy

Avoid the phrase "temporary view" for `data/shogi/datasets/<name>/`.

Use "Training View" or "Dataset Snapshot" for durable fixed views. Use
"run-local dataset" for disposable files under `runs/.../`.

## Acceptance Criteria

This issue can close when docs and scripts consistently distinguish durable
Training Views from run-local datasets, and no project guidance implies that
`data/shogi/datasets/<name>/` is temporary.

## Non-Goals

- define Training View directory naming policy
- add a Training View registry
- redesign Experience Store
- change current shogi `DatasetDefinition` schema
