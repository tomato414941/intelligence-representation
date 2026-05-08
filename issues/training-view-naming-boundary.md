# Training View Naming Boundary

Status: open.

## Issue

`Training View` may not be the best name for the current artifact.

The artifact is not a virtual view over source records. It is materialized on
disk and contains files such as:

- `train-games.jsonl`
- `eval-games.jsonl`
- `data-selection.json`
- `manifest.json`

That makes it closer to a fixed training data bundle, selected record set, or
materialized training slice.

## Why It Matters

The name should make the artifact's responsibility clear:

- it is not source data
- it is not a PyTorch `Dataset`
- it is not necessarily a virtual view
- it is a fixed input prepared for training / evaluation

If the name stays `Training View`, docs should clearly explain that it is a
materialized fixed view, not only a reference.

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

- decide whether Training View should be generalized
- redesign the shogi Training View format
- introduce a PyTorch Dataset abstraction
