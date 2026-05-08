# Training Data Bundle Naming Boundary

Status: closed.

## Issue

`Training View` was not the best name for the current artifact.

The artifact is not a virtual view over source records. It is materialized on
disk and contains files such as:

- `train-games.jsonl`
- `eval-games.jsonl`
- `data-selection.json`
- `manifest.json`

That made it closer to a fixed training data bundle, selected record set, or
materialized training slice than to a virtual view.

## Why It Matters

The name should make the artifact's responsibility clear:

- it is not source data
- it is not a PyTorch `Dataset`
- it is not necessarily a virtual view
- it is a fixed input prepared for training / evaluation

The chosen name is `Training Data Bundle`.

## Questions

- Should the name remain `Training View`?
- Would `Training Data Bundle`, `Training Record Set`, or `Training Slice` be
  clearer?
- Should the name emphasize records, samples, or training/evaluation use?

## Acceptance Criteria

- Decide whether to keep or rename `Training View`.
- If the name is kept, define it clearly in the glossary.
- If the name changes, update code, scripts, docs, and issue references without
  keeping compatibility aliases.

## Non-Goals

- decide whether Training Data Bundle should be generalized
- redesign the shogi Training Data Bundle format
- introduce a PyTorch Dataset abstraction

## Resolution

Renamed `Training View` to `Training Data Bundle` without compatibility aliases.

Updated code-level names include:

- `src/intrep/worlds/shogi/training_data_bundle.py`
- `scripts/create_shogi_training_data_bundle.py`
- `create_shogi_training_data_bundle`
- `shogi_training_data_bundle_v1`

The glossary now defines Training Data Bundle as a materialized, fixed
collection of training and evaluation inputs derived from source records.
