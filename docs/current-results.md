# Current Results

## Position

This document records the current prototype result after retiring the old experiment tree.

The project has not produced a latent world model or Transformer predictor. What exists now is a small action-conditioned prediction benchmark in `src/intrep`.

Historical experiment code and notes live under:

```text
legacy/experiments/
legacy/tests/
docs/legacy/
```

## Prototype Surface

```text
intrep.types:
  Fact / Action / Predictor

intrep.environment:
  MiniTransitionEnvironment

intrep.predictors:
  RuleBasedPredictor / FrequencyTransitionPredictor

intrep.evaluation:
  evaluate_prediction_cases

intrep.benchmark:
  run_benchmark

intrep.update_loop:
  PredictionErrorUpdateLoop
```

## Current Benchmark

The canonical executable result is `intrep.benchmark.run_benchmark()`.

It checks:

```text
rule baseline:
  predicts only hand-coded place actions

frequency predictor:
  learns transition outcomes from action-conditioned examples

prediction error update:
  adds an unsupported case to training memory and refits
```

Expected current result:

```text
train_size=6
test_size=6
rule_accuracy=0.17
frequency_accuracy=1.00
prediction_error=unsupported
update_success=True
training_size=6->7
```

## What This Shows

```text
small environment-generated data can beat a hand-written rule baseline
an unsupported case can become predictable after prediction-error update
```

## What This Does Not Show

```text
latent state
Transformer-based prediction
learned representation updates
large data
noise
partial observation
out-of-distribution generalization
planning or control
```

The current milestone is not "a world model is built."

It is:

```text
the repository now has a small installable prediction prototype with a benchmark
```

## Next Pressure

Next work should not add new taxonomies or experiment files.

It should either:

```text
1. expand generated data with held-out action / held-out object evaluation
2. replace the frequency baseline with a sequence or vector predictor
```
