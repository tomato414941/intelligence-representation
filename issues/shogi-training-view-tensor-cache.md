# Shogi Training View Tensor Cache

Status: open.

## Issue

Shogi policy-value training still rebuilds examples from Training View JSONL
files on each run.

The ShogiGameRecord source and Training View snapshot should stay as the
auditable source format, but repeated training should not always need to parse
JSONL, regenerate legal-move features, and materialize a large Python object
list.

## Why It Matters

Training views now give a fixed dataset input for training, but each run still
loads JSONL, rebuilds policy-value examples, and materializes Python objects.
This is simple, but it will become slow and memory-heavy as RL-generated shogi
experience grows.

## Current Context

The old Modal-based policy-value example cache route was removed when shogi
policy-value training moved to reading `ShogiGameRecord` data-selection sources
directly. That cleanup reduced the risk of treating `ShogiMoveChoiceExample`
JSONL as a second source of truth, but it did not remove the rebuild cost.

`ShogiGameRecord` JSONL is source-derived processed data, not a cache. A future
cache should be a regenerable acceleration artifact derived from a Training
View or source-derived records.

Old local cache artifacts still exist under `runs/shogi/`, including
`qhapaq-*-move-choice-examples.jsonl` and compressed `.zst` files. These are
legacy artifacts from the removed route and should not define the future cache
location.

## Initial Policy

Add a derived cache for shogi Training Views when JSONL rebuild cost becomes a
real blocker. Do not change the ShogiGameRecord source schema for this.

If the cache is added, keep it separate from source-derived records. A location
such as `data/qhapaq/cache/shogi-policy-value/` or a Training View-specific
cache directory is preferable to `runs/`, because reusable caches are not
run-output artifacts.

Prefer PyTorch mechanisms for runtime sampling and batching:

- `Dataset`
- `ConcatDataset`
- `Sampler` / `WeightedRandomSampler`
- `DataLoader`

## Acceptance Criteria

This issue can close when shogi Training Views have a PyTorch-native tensorized
cache or Dataset path that avoids rebuilding all policy-value examples from
JSONL each run.

## Non-Goals

- introduce a generic ExperienceStore abstraction
- define a shared store/view interface for other tasks
- replace ShogiGameRecord as the auditable source format
