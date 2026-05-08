# Shogi Training Data Bundle Source Mix

Status: closed.

## Issue

Shogi Experience Store can now contain multiple actor-pair sources, such as:

- `yaneuraou:yaneuraou`
- `checkpoint:yaneuraou`
- `yaneuraou:checkpoint`
- future `checkpoint:checkpoint`

Training Data Bundle creation now takes explicit train/eval game logs instead of
splitting the full store directly. That makes heldout evaluation sources
possible, but it does not yet solve source mix selection: the caller still has
to produce those train/eval logs intentionally.

## Why It Matters

Experience Store should keep generated shogi experience. Training Data Bundle should
define what a model trains on for a specific run.

If Training Data Bundle source mix is not visible, experiments may accidentally train
on "whatever is currently in the store" without noticing the actual composition.

## Initial Policy

Do not add source-mix controls yet. Keep selection manual until a concrete
experiment needs actor-pair include/exclude or caps.

If controls are later needed, keep the first implementation small. Possible
examples:

- cap `checkpoint:yaneuraou` games while keeping more `yaneuraou:yaneuraou`
  games
- exclude a weak self-play batch from evaluation
- exclude or cap known weak or obsolete actor-pair slices
- compare two explicit source mixes from the same store

## Acceptance Criteria

This issue can close when Training Data Bundle manifests clearly record source mix,
and source-mix selection remains manual until a concrete experiment needs
controls.

## Resolution

Source mix control is not needed yet. The current requirement is observability:
a Training Data Bundle must show what source mix it contains.

Current manifests already record this with:

- `actor_pair_counts`
- `train_actor_pair_counts`
- `eval_actor_pair_counts`

The current active view records:

```json
{
  "actor_pair_counts": {
    "checkpoint:yaneuraou": 165,
    "yaneuraou:checkpoint": 165,
    "yaneuraou:yaneuraou": 152
  },
  "train_actor_pair_counts": {
    "checkpoint:yaneuraou": 124,
    "yaneuraou:checkpoint": 124,
    "yaneuraou:yaneuraou": 114
  },
  "eval_actor_pair_counts": {
    "checkpoint:yaneuraou": 41,
    "yaneuraou:checkpoint": 41,
    "yaneuraou:yaneuraou": 38
  }
}
```

Future include/exclude or cap controls should be added only when a concrete
experiment needs them.
