# Shogi Training Data Bundle Tensor Cache

Status: closed.

## Issue

Shogi policy-value training could rebuild examples from Training Data Bundle JSONL
files on each run.

The ShogiGameRecord source and Training Data Bundle snapshot should stay as the
auditable source format, but repeated training should not always need to parse
JSONL, regenerate legal-move features, and materialize a large Python object
list.

## Why It Matters

Training data bundles now give a fixed dataset input for training, but each run still
loads JSONL, rebuilds policy-value examples, and materializes Python objects.
This is simple, but it will become slow and memory-heavy as RL-generated shogi
experience grows.

## Resolution

Closed by adding a shared tensorized shogi policy/value sample representation
and a bundle-local tensor cache path.

- `CandidateMovePolicyValueTensorSample` is the runtime training sample.
- `ShogiPolicyValueDataset` accepts either semantic `ShogiPolicyValueExample`
  objects or tensorized samples.
- Online Replay stores tensorized samples in its `ReplayBuffer`.
- `scripts/build_shogi_policy_value_tensor_cache.py` builds a cache from a
  Training Data Bundle `data-selection.json`.
- `intrep.train_shogi_policy_value --tensor-cache ...` trains from the cache and
  rejects caches whose embedded data selection does not match the requested
  selection.

The ShogiGameRecord source and Training Data Bundle snapshot remain the
auditable source format. The tensor cache is a rebuildable acceleration artifact
under the bundle's `cache/` directory.

## Cache Policy

The cache must stay separate from source-derived records. Reusable caches belong
under the Training Data Bundle directory, not under `runs/`.

## Acceptance Criteria

This issue can close when shogi Training Data Bundles have a PyTorch-native tensorized
cache or Dataset path that avoids rebuilding all policy-value examples from
JSONL each run.

Met.

## Non-Goals

- introduce a generic ExperienceStore abstraction
- define a shared store/view interface for other tasks
- replace ShogiGameRecord as the auditable source format
