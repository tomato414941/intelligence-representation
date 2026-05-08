# Mixed-Schema Dataset Boundary

Status: open.

## Issue

The project now defines `Sample Schema`, but it has not decided how mixed-schema
datasets should work.

A dataset can contain one Sample Schema or multiple Sample Schemas. This matters
because the project wants to learn across image, text, grid, shogi, and future
experience sources without forcing all data into one narrow shape.

## Why It Matters

Mixed-schema learning is close to the project goal, but it can become unclear
which boundary is actually unified:

- an envelope schema only unifies the outer container, such as `schema_id` plus
  payload
- a unified core input schema unifies the representation passed into the shared
  core
- a unified sample schema unifies the sample fields and meanings themselves

These are different claims. Treating them as the same would make model routing,
collation, losses, metrics, and cache ownership hard to reason about.

## Direction

Do not introduce a generic mixed-schema framework yet.

When this becomes concrete, first decide:

- whether the dataset is single-schema or mixed-schema
- whether batches are schema-homogeneous or mixed-schema
- whether unification happens at the envelope, shared core input, or full sample
  schema boundary
- which input embedding modules and output heads remain schema-specific
- how losses and metrics are computed and reported per schema

Core-only or core-focused training from precomputed unified core inputs may be a
valid speed strategy, but it should not be described as end-to-end unification
of input embedding modules and output heads.

## Acceptance Criteria

- Decide whether the first mixed-schema experiment should use schema-homogeneous
  batches or mixed-schema batches.
- Decide whether the first unification target is envelope schema, unified core
  input schema, or unified sample schema.
- Document the chosen boundary before implementing shared routing or collation.

## Non-Goals

- rename `tasks/`
- implement a generic multi-task trainer now
- force all data into one token or sequence representation
