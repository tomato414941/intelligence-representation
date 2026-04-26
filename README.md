# intelligence-representation

This repository explores representation for intelligence through an installable, testable prototype.

The current center is not a hand-designed semantic database. It is a minimal world-model-style prediction loop:

```text
Observation
  -> Prediction
  -> Action / Query
  -> New Observation
  -> Prediction Error / Update
```

The active implementation is `src/intrep`. Historical experiments are kept under `legacy/` for reference only.

## Current Position

The current implementation is a toy symbolic prototype. It does not yet contain a latent world model. It now includes a tiny trained Transformer predictor over symbolic world-model tokens.

Its narrower purpose is:

```text
Can action-conditioned training data and prediction-error updates
change next-state prediction outcomes in measurable ways?
```

The default benchmark compares a hand-written rule baseline, a frequency-based transition predictor, a state-aware predictor, a Transformer-ready sequence adapter, a dependency-free sequence-feature baseline, and a tiny trained Transformer. It breaks out seen patterns, held-out objects, longer chains, missing links, and noisy distractors, then checks whether an unsupported case can be corrected by adding the prediction error to training memory.

## Project Map

```text
src/intrep/
  Active prototype package

tests/
  Default test suite for src/intrep

legacy/experiments/
  Retired experiment code

legacy/tests/
  Retired experiment tests

docs/legacy/
  Historical experiment notes and older architecture sketches
```

Current implementation surface:

```text
src/intrep/types.py
src/intrep/dataset.py
src/intrep/environment.py
src/intrep/predictors.py
src/intrep/evaluation.py
src/intrep/tokens.py
src/intrep/sequence.py
src/intrep/sequence_predictor.py
src/intrep/torch_sequence.py
src/intrep/tiny_transformer.py
src/intrep/benchmark.py
src/intrep/update_loop.py
```

## Canonical Docs

Read these first:

```text
docs/world-model.md
docs/bitter-lesson.md
docs/evaluation.md
docs/current-results.md
```

Broad background:

```text
docs/concept.md
```

Legacy / exploratory notes:

```text
docs/legacy/
```

## Design Constraints

This repository should avoid turning into a handcrafted ontology project.

Prefer:

```text
small prediction tasks
clear baselines
measured prediction accuracy
condition-level breakdowns
predictor comparisons that expose failures
small trained Transformer benchmarks before scaling model size
prediction error updates
held-out cases before new abstractions
```

Avoid adding new broad taxonomies, fixed schemas, or semantic dataclasses unless an experiment repeatedly forces them.

## Run Tests

```sh
uv sync
uv run python -m unittest
```

## Run Demo

```sh
uv run python -m intrep.demo
```
