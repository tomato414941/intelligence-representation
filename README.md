# intelligence-representation

This repository explores representation for intelligence through small, testable experiments.

The current center is not a hand-designed semantic database. It is a minimal world-model loop:

```text
Observation
  -> Prediction
  -> Action / Query
  -> New Observation
  -> Prediction Error / Update
```

The project treats raw observations as the source of truth. Derived summaries, beliefs, conflicts, and symbolic facts are temporary views or evaluation artifacts, not the final architecture.

## Current Position

The current implementation is a toy symbolic environment. It does not yet contain a learned world model, latent state, or Transformer predictor.

Its purpose is narrower:

```text
Can memory, retrieval, multi-hop context, time, conflict, and reliability
change next-state prediction outcomes in measurable ways?
```

Experiments 17-22 are the current main line:

```text
17: Observation-assisted prediction
18: Multi-hop observation prediction
19: Ambiguous multi-hop prediction
20: Temporal multi-hop prediction
21: Temporal conflict prediction
22: Reliability-weighted prediction
```

Earlier semantic/state-memory experiments are historical concept sketches. They are kept for context, but they are not the current architecture.

## Canonical Docs

Read these first:

```text
docs/world-model.md
docs/bitter-lesson.md
docs/evaluation.md
```

Broad background:

```text
docs/concept.md
```

Legacy / exploratory notes:

```text
docs/architecture.md
docs/state-taxonomy.md
docs/failure-modes.md
docs/experiment-01-*.md ... docs/experiment-16-*.md
```

## Design Constraints

This repository should avoid turning into a handcrafted ontology project.

Prefer:

```text
small prediction tasks
clear baselines
measured prediction accuracy
context size and retrieval cost
provenance and counterevidence tracking
```

Avoid adding new broad taxonomies, fixed schemas, or semantic dataclasses unless an experiment repeatedly forces them.

## Run Tests

```sh
python3 -m unittest
```

