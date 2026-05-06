# Shogi Experience Store PyTorch Dataset Boundary

Status: open.

## Issue

The current experience-store and training-view workflow is shogi-specific and
still feeds training through ShogiGameRecord JSONL files.

This is acceptable as the first concrete implementation, but it does not yet
define when to introduce PyTorch-native dataset/cache/sampler pieces, or when
similar workflows in other tasks should share a common interface.

## Why It Matters

Training views now give a fixed dataset input for training, but each run still
loads JSONL, rebuilds move-choice examples, and materializes Python objects.
This is simple, but it will become slow and memory-heavy as RL-generated shogi
experience grows.

At the same time, making a generic experience-store abstraction now would be
premature unless image, grid, text, or another task needs the same lifecycle.

## Initial Policy

Keep the experience store shogi-specific until at least one more concrete task
needs the same lifecycle:

- append source records into a mutable experience store
- create an immutable training view
- train through a PyTorch Dataset/DataLoader

When performance becomes the blocker, add a derived tensorized cache for shogi
training views rather than changing the ShogiGameRecord source schema.

Prefer PyTorch mechanisms for runtime sampling and batching:

- `Dataset`
- `ConcatDataset`
- `Sampler` / `WeightedRandomSampler`
- `DataLoader`

## Acceptance Criteria

This issue can close when one of the following is true:

- shogi training views have a PyTorch-native tensorized cache or Dataset path
  that avoids rebuilding all examples from JSONL each run
- a second concrete task needs the same store/view lifecycle and a minimal
  shared interface is introduced from both use cases
