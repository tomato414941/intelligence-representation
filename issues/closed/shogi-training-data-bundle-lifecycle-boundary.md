# Shogi Training Data Bundle Lifecycle Boundary

Status: closed.

## Issue

The project should clearly distinguish durable Training Data Bundles from run-local
datasets.

A view under `data/shogi/training-data-bundles/<name>/` is not temporary; it is a fixed,
reusable snapshot. A dataset written under `runs/.../` is run-local and
disposable.

Calling both "temporary views" makes the storage boundary unclear.

## Why It Matters

`runs/` should be disposable experiment output. If run-local datasets and
durable Training Data Bundles are described with the same name, it becomes unclear
which files can be deleted and which files are intended to be reused.

The distinction should be simple:

- durable fixed view: `data/shogi/training-data-bundles/<name>/`
- run-local dataset: `runs/.../dataset.json` and adjacent source files

## Initial Policy

Avoid the phrase "temporary view" for `data/shogi/training-data-bundles/<name>/`.

Use "Training Data Bundle" or "Dataset Snapshot" for durable fixed views. Use
"run-local dataset" for disposable files under `runs/.../`.

## Acceptance Criteria

This issue can close when docs and scripts consistently distinguish durable
Training Data Bundles from run-local datasets, and no project guidance implies that
`data/shogi/training-data-bundles/<name>/` is temporary.

## Resolution

No problematic "temporary view" wording was found outside this issue.

The current boundary is:

- `data/shogi/training-data-bundles/<name>/`: durable shogi Training Data Bundle / Dataset Snapshot.
- `runs/.../dataset.json` and adjacent source files: run-local dataset.

`create_shogi_training_data_bundle.py` defaults to `data/shogi/training-data-bundles/` and refuses
to overwrite an existing view, which matches the durable Training Data Bundle
interpretation.

The placement rule is recorded in
[`../../docs/artifact-layout.md`](../../docs/artifact-layout.md).

## Non-Goals

- define Training Data Bundle directory naming policy
- add a Training Data Bundle registry
- redesign Experience Store
- change current shogi `DatasetDefinition` schema
