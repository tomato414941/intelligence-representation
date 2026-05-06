# Shogi Training View Source Mix

Status: open.

## Issue

Shogi Experience Store can now contain multiple actor-pair sources, such as:

- `yaneuraou:yaneuraou`
- `checkpoint:yaneuraou`
- `yaneuraou:checkpoint`
- future `checkpoint:checkpoint`

Training View creation now takes explicit train/eval game logs instead of
splitting the full store directly. That makes heldout evaluation sources
possible, but it does not yet solve source mix selection: the caller still has
to produce those train/eval logs intentionally.

## Why It Matters

Experience Store should keep generated shogi experience. Training View should
define what a model trains on for a specific run.

If Training View cannot choose source mix, experiments may accidentally train on
"whatever is currently in the store" rather than an intentional dataset.

## Initial Policy

Do not add source-mix controls yet. The current store does not have enough
`yaneuraou:yaneuraou` or self-play data to justify the extra interface.

Add controls only when at least one real experiment needs a deliberate mix, for
example:

- cap `checkpoint:yaneuraou` games while keeping more `yaneuraou:yaneuraou`
  games
- exclude a weak self-play batch from evaluation
- compare two explicit source mixes from the same store

## Acceptance Criteria

This issue can close when Training View creation can express the needed source
mix without making Training depend on run-output directories.

The first implementation should stay simple, likely actor-pair include/exclude
or per-actor-pair game caps. Avoid weighted sampling or generic dataset
composition until a concrete run requires it.
