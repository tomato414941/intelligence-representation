# Learning Data Boundaries

This document records design rules for learning data boundaries. The glossary
is the source of truth for term definitions.

## Scope

This document is about the boundary between source-side material and training
consumption.

It covers where learning data is selected, fixed, adapted, and consumed. It does
not define model architecture, domain runtime, or a generic reinforcement
learning framework.

## Source-Side Material

Source-side material should remain usable before an objective, model, or run is
chosen.

Do not reshape source-side material only to fit one objective. The same material
may later feed different target construction rules, training examples,
evaluators, or diagnostics.

Do not force static dataset records and interaction records into one generic raw
schema. They are related because both can feed learning data, not because they
need the same stored fields.

## Selection Boundary

Data Selection is the inclusion and split boundary:

```text
source-side material
  -> Data Selection
  -> selected material with split assignment
```

Data Selection decides what existing records or stored source-side material are
included for a declared use, and how that material is assigned to splits.

It should not perform target construction, training-example construction,
runtime sampling, optimization, or metric computation.

## Fixed Training Input

A Training Data Bundle is the fixed training input built from a declared Data
Selection.

```text
Data Selection
  -> Training Data Bundle
```

The bundle is not source data and not a PyTorch `Dataset`. It is a materialized
training artifact that lets a run consume fixed inputs without relying on
run-local generated files as the source of truth.

## Dataset Boundary

PyTorch `Dataset` objects are adapters over already-selected or bundled
material.

```text
Training Data Bundle
  -> PyTorch Dataset
  -> indexed samples
```

A Dataset should not own source storage, split assignment, target construction,
or learning intent. It should adapt a selected training or evaluation set into
runtime samples.

## Experience Storage

Experience Stores are source storage for generated or collected experience.
They are not PyTorch Datasets and should not be consumed directly by training.

Training should consume a declared Data Selection, or a fixed Training Data
Bundle built from one.

## Run Artifacts

A run produces artifacts such as raw logs, caches, metrics, and checkpoints.
Runs can provide material for future Data Selection, but a run artifact path is
not itself the selection boundary.

Training should be explainable by Data Selection, training-example meaning, and
training configuration, not only by a convenient artifact path.

## Target Construction

Target Construction is downstream of selection.

```text
selected material
  -> Target Construction
  -> Training Example
```

Selection decides what material is included. Target Construction decides how
included material becomes targets, feedback, or objective-specific training
examples.

## Domain-Specific Boundaries

Domain-specific learning and runtime artifact boundaries belong under the
domain documentation, such as `docs/shogi/`.

## Non-Goals

This document does not introduce:

```text
a universal raw-data envelope
a generic dataset schema
a generic target-construction framework
a generic reinforcement-learning runtime
a model-architecture boundary
a required agent loop
```

Add those only when experiments require them.
