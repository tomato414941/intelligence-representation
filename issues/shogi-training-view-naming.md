# Shogi Training View Naming

Status: open.

## Issue

Shogi Training Views are starting to encode too much experiment detail in their
directory names.

Recent examples include names such as:

- `current-with-heldout-yaneuraou-eval`
- `current-g100-heldout-yaneuraou-g20`

These names are useful while debugging, but they can become hard to manage once
views are materialized often. The name also starts to duplicate information that
already belongs in `dataset.json` and `manifest.json`, such as source paths,
`max_games`, actor pairs, and target policy.

There is also a naming mismatch between durable views and run-local temporary
datasets. A view under `data/shogi/datasets/<name>/` is not temporary; it is a
fixed, reusable snapshot. A dataset written under `runs/.../` is run-local and
disposable. Calling both "temporary views" makes the storage boundary unclear.

## Why It Matters

Training View names should identify the view without becoming the source of
truth for the experiment. The source of truth should remain the dataset
definition and manifest.

If names carry too much meaning, users may compare or select views by parsing
names instead of reading the recorded metadata.

## Initial Policy

Do not solve this yet with a registry or numbered sequence.

Avoid the phrase "temporary view" for `data/shogi/datasets/<name>/`.
Use "Training View" or "Dataset Snapshot" for durable fixed views, and
"run-local dataset" for disposable files under `runs/.../`.

When this becomes annoying in practice, choose a small naming rule that keeps
names short and leaves details in metadata. Candidate directions:

- one stable `current` view for the active local view
- short human names plus manifest metadata
- a date or short slug only when multiple materialized views must coexist

Avoid names that try to encode the whole source mix, target policy, and limits.

## Acceptance Criteria

This issue can close when the project has a simple Training View naming policy,
clearly distinguishes durable fixed views from run-local datasets, and
`create_shogi_training_view.py` usage follows it without introducing a separate
view registry.
