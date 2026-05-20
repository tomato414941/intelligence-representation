# Tensor Cache Input Output Boundary

Status: open. Priority: low.

## Issue

Shogi policy/value tensor caches are currently training-ready sample caches.
They combine input-side tensors and output-side targets in one artifact:

- position features
- pair relation edges
- policy targets
- legal masks
- value targets

This is simple and appropriate for the current full training path, but it makes
input-side model-line comparisons more expensive than necessary. For example,
removing one input-side relation such as `PAIR_RELATION_PIECE_SAME_SIDE`
requires rebuilding the whole tensor sample cache even though the policy/value
targets did not change.

## Current Policy

Keep the current unified tensor sample cache for now.

Do not split input feature caches and output target caches only because the
separation is possible. Split them only when repeated model-line comparisons or
multiple output spaces make the rebuild cost or artifact duplication a real
bottleneck.

## Future Shape

A future design may separate:

- input feature cache: input schema and feature manifest dependent
- output target cache: output space and target schema dependent
- tensor sample view: alignment layer that joins input and target artifacts by
  stable example identity

The alignment layer is the critical part. Input shard row `N` and target shard
row `N` must refer to the same Training Data Bundle example, or training can
silently corrupt.

## Trigger

Revisit when the project repeatedly compares model entries with different input
representations or output spaces from the same Training Data Bundle and cache
rebuild time becomes a meaningful constraint.

## Non-Goals

- block policy-plane tensor cache creation
- redesign Training Data Bundle
- add generic cache abstractions before a concrete second need exists
