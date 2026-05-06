# Shogi Training View Tensor Cache

Status: open.

## Issue

Shogi move-choice training still rebuilds examples from Training View JSONL
files on each run.

The ShogiGameRecord source and Training View snapshot should stay as the
auditable source format, but repeated training should not always need to parse
JSONL, regenerate legal-move features, and materialize a large Python object
list.

## Why It Matters

Training views now give a fixed dataset input for training, but each run still
loads JSONL, rebuilds move-choice examples, and materializes Python objects.
This is simple, but it will become slow and memory-heavy as RL-generated shogi
experience grows.

## Initial Policy

Add a derived cache for shogi Training Views when JSONL rebuild cost becomes a
real blocker. Do not change the ShogiGameRecord source schema for this.

Prefer PyTorch mechanisms for runtime sampling and batching:

- `Dataset`
- `ConcatDataset`
- `Sampler` / `WeightedRandomSampler`
- `DataLoader`

## Acceptance Criteria

This issue can close when shogi Training Views have a PyTorch-native tensorized
cache or Dataset path that avoids rebuilding all move-choice examples from
JSONL each run.

## Non-Goals

- introduce a generic ExperienceStore abstraction
- define a shared store/view interface for other tasks
- replace ShogiGameRecord as the auditable source format
