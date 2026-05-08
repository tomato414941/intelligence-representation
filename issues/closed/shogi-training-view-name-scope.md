# Shogi Training View Name Scope

Status: closed.

## Issue

Shogi Training View directory names are starting to encode too much experiment
detail.

Recent examples include names such as:

- `current-with-heldout-yaneuraou-eval`
- `current-g100-heldout-yaneuraou-g20`

These names are useful while debugging, but they can become hard to manage once
views are materialized often. The name also starts to duplicate information that
already belongs in `dataset.json` and `manifest.json`, such as source paths,
`max_games`, actor pairs, and target policy.

## Why It Matters

Training View names should identify the view without becoming the source of
truth for the experiment. The source of truth should remain the dataset
definition and manifest.

If names carry too much meaning, users may compare or select views by parsing
names instead of reading the recorded metadata.

## Initial Policy

Do not solve this yet with a registry or numbered sequence.

When this becomes annoying in practice, choose a small naming rule that keeps
names short and leaves details in metadata. Candidate directions:

- one stable `current` view for the active local view
- short human names plus manifest metadata
- a date or short slug only when multiple materialized views must coexist

Avoid names that try to encode the whole source mix, target policy, and limits.

## Acceptance Criteria

This issue can close when the project has a simple Training View directory name
policy and `create_shogi_training_view.py` usage follows it without introducing
a separate view registry.

## Resolution

The local Training View set was reduced to one active durable view:

- `data/shogi/datasets/current/`

The policy is:

- keep `current/` as the normal active Training View
- use `runs/` for temporary views
- add another durable Training View only when it has a concrete reuse reason
- keep details such as source mix, target policy, limits, and actor pairs in
  `dataset.json` and `manifest.json`, not in the directory name

This avoids a registry or numbered sequence for now.

The placement rule is recorded in
[`../../docs/artifact-layout.md`](../../docs/artifact-layout.md).
