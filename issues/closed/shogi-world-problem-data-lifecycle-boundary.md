# Shogi World / Problem Data Lifecycle Boundary

Status: closed.

## Issue

The boundary between `intrep.worlds.shogi` and
`intrep.problems.shogi_policy_value` is improved but still easy to blur.

Current nearby concepts include:

- ShogiGameRecord
- Experience Store
- Training Data Bundle
- Data Selection
- tensor cache
- policy/value examples and samples
- generated-data and Online Replay loops

Some of these are source-side shogi world artifacts. Others are
policy/value-problem training inputs. If the boundary stays implicit, future
changes may place data lifecycle code wherever it is convenient at the moment.

## Desired Direction

Keep the conceptual split explicit:

- `worlds/shogi/` owns shogi source-side records, source-derived stores,
  Training Data Bundle creation, and shogi-world data formats.
- `problems/shogi_policy_value/` owns conversion from selected shogi data into
  policy/value samples, tensorized runtime samples, model training, evaluation,
  and learning loops for that problem.

This issue should clarify the boundary only when a concrete file placement or
responsibility conflict appears. Do not introduce a generic world/problem data
framework from shogi alone.

## Acceptance Criteria

This issue can close when the current shogi data lifecycle files have clear
documented ownership, and there is no active ambiguity about where new shogi
source-side versus policy/value-problem code belongs.

Met.

## Resolution

The boundary is documented in `docs/shogi-learning-boundaries.md`, with a short
link from `docs/learning-boundaries.md`.

The documented rule is source-side versus problem-side:

- `intrep.worlds.shogi` owns source-side shogi data and formats.
- `intrep.problems.shogi_policy_value` owns policy/value targets, samples,
  tensor caches, model training/evaluation, and learner loops.

## Non-Goals

- redesign Training Data Bundle format
- redesign Experience Store
- generalize Training Data Bundle to other worlds now
- generalize Experience Store to other worlds now
