# Shogi Policy/Value Example Sample Boundary

Status: open. Priority: medium.

## Issue

`src/intrep/problems/shogi_policy_value/examples.py` mixes several concepts:

- durable training examples
- JSONL read/write for those examples
- runtime tensor samples
- example-to-sample tensorization

This makes later changes harder to place cleanly. For example, tensor-cache
storage dtype compaction belongs to cache/sample serialization, not to durable
training examples.

## Desired Shape

Separate the problem-side concepts by responsibility:

- `examples.py`
  - `ShogiMovePolicyValueExample`
  - durable JSONL read/write
- `samples.py`
  - runtime tensor sample dataclasses
  - sample validation/conversion helpers that are independent of durable JSONL
- `tensorization.py`
  - conversion from durable examples to runtime samples
  - legal-move and policy-plane tensorization paths

`tensor_cache.py` should depend on samples/tensorization, but durable example
records should not know tensor-cache storage details.

## Non-Goals

- change the training example schema
- change policy/value target semantics
- change input or output representations
- introduce a generic cross-problem example/sample framework

## Notes

This is a boundary cleanup. It should make storage dtype compaction and future
policy/output variants easier to implement without making `examples.py` a
catch-all module.
