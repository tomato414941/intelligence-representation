# intelligence-representation

This repository explores representation for intelligence through a small, testable prototype.

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

The current implementation is a toy symbolic prototype. It does not yet contain a latent world model or Transformer predictor.

Its purpose is narrower:

```text
Can memory, retrieval, multi-hop context, time, conflict, and reliability
change next-state prediction outcomes in measurable ways?
```

The prototype lives in `src/intrep/`.

```text
src/intrep/types.py
src/intrep/dataset.py
src/intrep/environment.py
src/intrep/predictors.py
src/intrep/evaluation.py
src/intrep/update_loop.py
```

Experiments 17-24 were the path that led to the prototype:

```text
17: Observation-assisted prediction
18: Multi-hop observation prediction
19: Ambiguous multi-hop prediction
20: Temporal multi-hop prediction
21: Temporal conflict prediction
22: Reliability-weighted prediction
23: Learned transition predictor
24: Prediction error update loop
```

Earlier semantic/state-memory experiments are historical concept sketches. They are kept for context, but they are not the current architecture.

## Experiment Map

`src/intrep/` is the current implementation surface.
`experiments/` is now legacy exploration plus runnable compatibility demos.

Prototype wrappers:

```text
experiments/learned_transition_predictor.py
experiments/prediction_error_update_loop.py
```

Earlier current-line experiments kept as exploration:

```text
experiments/observation_assisted_prediction.py
experiments/multihop_observation_prediction.py
experiments/ambiguous_multihop_prediction.py
experiments/temporal_multihop_prediction.py
experiments/temporal_conflict_prediction.py
experiments/reliability_weighted_prediction.py
```

Legacy concept sketches:

```text
experiments/semantic_state.py
experiments/contextual_claims.py
experiments/contextual_state_update.py
experiments/observation_belief_conflict.py
experiments/semantic_memory.py
experiments/observation_stream.py
experiments/world_state_update.py
experiments/state_hub.py
experiments/observation_memory_log.py
experiments/ngram_observation_memory.py
experiments/retrieval_evaluation.py
experiments/predictive_loop.py
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
python3 -m venv .venv
./.venv/bin/python -m pip install -e .
./.venv/bin/python -m unittest
```

## Run Demo

```sh
./.venv/bin/python -m intrep.demo
```
