# Learning Data Responsibility Boundaries

Status: closed.

## Issue

The project needs clearer responsibilities between source data and training.
Naming should follow responsibility, not the other way around.

This matters because names such as `Dataset Definition`, `Training Data Bundle`, and
`Example` can become too broad if they are assigned before the responsibilities
are separated.

## Responsibility Draft

Separate these responsibilities before promoting names into glossary terms:

- source storage: preserve source-side records without reshaping them around
  one objective, model, or run
- Data Selection: decide which source records, training examples, or stored
  targets are included for a declared use
- Training Example Definition: define how included data is treated as
  input/target relationships for objective-specific training examples
- runtime sampling: adapt training examples into PyTorch samples
- optimization: turn objectives into losses or learning signals
- artifact storage: store run outputs, caches, checkpoints, and metrics

## Current Concern

`Dataset Definition` may be too broad as a name if it absorbs Data Selection,
Training Example Definition, target availability, sampling, or training
configuration.

`Data Selection` is the current candidate name for the responsibility of
deciding what data is included for a declared use. The declared use may be
training, evaluation, target generation, analysis, comparison, or retrieval
indexing. Data Selection should not decide how targets are generated, how
examples are constructed, how samples are batched, or what objective/loss is
optimized.

`Training Example Definition` is the current candidate name for the
responsibility of defining how selected data becomes objective-specific
training examples. It may include input/target roles, lightweight source-to-input
conversion, target reference or derivation, and target shaping needed by the
example. Heavy external target generation, such as running an engine, search, a
teacher model, or human annotation, should remain outside this responsibility as
stored target or artifact work.

Training Example Definition should distinguish input/target roles and forms
from how those values are produced. Construction from existing source records or
stored targets is different from external target generation by engines, search,
teacher models, or human annotation.

This issue stops at model input and target construction. Input layers, shared
cores, hidden states, output heads, and model outputs belong to model-boundary
documents.

The first boundary to decide is responsibility, not the final name.

## Current Direction

Prefer `Data Selection` over `Dataset Definition` for the inclusion boundary.
`Dataset Definition` is too easy to read as "everything needed to define a
dataset", which can absorb example definition, target handling, sampling,
training configuration, or artifact paths.

Keep `Training Example Definition` as a separate candidate responsibility. It
is important, but its exact scope is still less stable than Data Selection.

Do not keep `DatasetDefinition` as an implementation name once the code boundary
is revisited. Do not introduce compatibility aliases when the rename or split
happens.

## Progress

- `Data Selection` has been promoted to `docs/glossary.md`.
- `Dataset Definition` has been removed from `docs/glossary.md` as a formal
  term.
- Code names have been moved away from `DatasetDefinition` toward Data
  Selection.
- `Training Example Definition` has not been promoted as a separate glossary
  term. Instead, `Training Example` now carries the input/target-or-feedback
  meaning directly.

## Resolution

The responsibility boundary is now stable enough for this issue:

- `Data Selection` owns inclusion for a declared use.
- `Training Example` owns the objective-specific input/target-or-feedback unit.
- PyTorch `Dataset` remains a runtime sample adapter.
- `Dataset Definition` is no longer a formal glossary term.

Remaining code-level mixing is tracked separately in
`training-example-responsibility-mixing.md`.

## Acceptance Criteria

- Decide which responsibilities need project-level names.
- Decide whether existing data-selection code should be split further into Data
  Selection and Training Example Definition responsibilities.
- Promote only stable terms into `docs/glossary.md`.
- Keep relationship explanations in `docs/learning-boundaries.md` only after
  the responsibilities are stable enough.

## Non-Goals

- introduce a generic dataset framework
- rename existing classes immediately
- redesign shogi training data storage in this issue
