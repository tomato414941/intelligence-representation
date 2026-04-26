# Evaluation

## Current Line

The current evaluation line is the `src/intrep` prototype, not the retired experiment tree.

The main question is:

```text
Does action-conditioned training data improve next-state prediction,
and can prediction-error updates make unsupported cases predictable?
```

The benchmark should also expose when a predictor succeeds by memorizing seen patterns, when it must use current state relations, and when unsupported is the correct output.

Past semantic-memory, retrieval, conflict, and state-taxonomy tests are historical sketches. They now live under `legacy/tests/`.

## Metrics

The current benchmark tracks:

```text
prediction accuracy
unsupported rate
condition-level accuracy
training size before / after update
update success
```

Human-readable Fact / Action structures are evaluation artifacts. They are not claimed to be the final architecture.

## Current Tests

```text
tests/test_benchmark.py:
  checks rule baseline vs frequency predictor vs state-aware predictor,
  condition-level failures,
  and prediction-error update success

tests/test_learned_transition_predictor.py:
  checks generated action-conditioned examples and learned predictor behavior

tests/test_prediction_error_update_loop.py:
  checks unsupported -> update -> correct behavior

tests/test_demo.py:
  checks the package demo runs and prints benchmark results

tests/test_intrep_imports.py:
  checks the current package surface imports normally
```

## Acceptance Criteria

A change is useful only if it improves or clarifies at least one of:

```text
next-state prediction accuracy
unsupported rate
held-out generalization
prediction-error update behavior
benchmark clarity
```

Avoid adding broad schemas, ontology categories, or new experiment files unless the benchmark exposes a repeated need for them.
