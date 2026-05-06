# Shogi Training View Source Mix

Status: open.

## Issue

Shogi Experience Store can now contain multiple actor-pair sources, such as:

- `yaneuraou:yaneuraou`
- `checkpoint:yaneuraou`
- `yaneuraou:checkpoint`
- future `checkpoint:checkpoint`

Training View creation currently snapshots the full store. That is fine while
the store is small, but it will become too coarse once one source dominates the
store or a run needs a deliberate teacher/student/self-play mix.

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
