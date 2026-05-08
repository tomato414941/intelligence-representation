# Shogi World Data Directory Boundary

Status: open.

## Issue

`data/shogi/` is currently acceptable as the local home for shogi generated
experience, Training Views, and copied record sets, but the boundary still feels
awkward.

The discomfort is that `shogi` is a world/domain name, not a concrete data
source name like `qhapaq`, `wikitext-2`, or `project-gutenberg`.

That means `data/shogi/` can naturally attract many unrelated shogi-shaped
things:

- generated self-play experience
- YaneuraOu-vs-YaneuraOu records
- model-vs-engine records
- Training Views
- copied record sets
- future position annotations

## Why It Matters

The current contents are explainable, but a world-level data bucket can keep
growing until it hides source, lifecycle, and responsibility boundaries.

The project should avoid both bad outcomes:

- prematurely splitting directories into abstract categories before a concrete
  need exists
- letting `data/shogi/` become the default place for every shogi-related file

## Current Policy

Leave the current layout in place for now.

The immediate `data/shogi/` content audit is closed in
[`closed/data-shogi-directory.md`](closed/data-shogi-directory.md). This issue
tracks only the broader naming and boundary discomfort.

## Questions

- Should shogi generated data eventually live under separate top-level
  directories such as `data/shogi-experiences/`, `data/shogi-training-views/`,
  and `data/shogi-records/`?
- Should source-first naming be preferred instead, such as `data/yaneuraou/` or
  `data/model-selfplay/`?
- Should world-level directories be allowed under `data/` only for generated
  experience and Training Views?
- What concrete pain would justify moving files out of `data/shogi/`?

## Acceptance Criteria

This issue can close when one of the following is true:

- the current `data/shogi/` world-level directory is explicitly accepted as the
  project policy, with clear limits on what may go there, or
- a concrete file move splits the shogi data layout into clearer source or
  lifecycle boundaries.

## Non-Goals

- move files immediately
- redesign Experience Store or Training View
- introduce a generic generated-data directory framework
