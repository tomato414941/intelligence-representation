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
  RuleBasedPredictor / FrequencyTransitionPredictor / StateAwarePredictor / TransformerReadyPredictor

intrep.tokens / intrep.sequence:
  world-model tokenization and sequence examples

intrep.sequence_predictor:
  dependency-free sequence feature baseline

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

state-aware predictor:
  uses current located_at relations as a fallback when frequency lookup fails

transformer-ready adapter:
  uses the same token sequence shape a trained Transformer predictor would consume,
  but it is not a trained Transformer yet

sequence-feature baseline:
  learns from token features without external ML dependencies,
  and exposes why a real sequence model is still needed

condition slices:
  expose seen patterns, held-out objects, longer chains, missing links, and noisy distractors

prediction error update:
  adds an unsupported case to training memory and refits
```

Expected current result:

```text
train_size=6
test_size=15
rule_accuracy=0.20
frequency_accuracy=0.53
state_aware_accuracy=1.00
transformer_ready_accuracy=0.53
sequence_feature_accuracy=0.40
seen_action_patterns.frequency_accuracy=1.00
seen_action_patterns.state_aware_accuracy=1.00
seen_action_patterns.transformer_ready_accuracy=1.00
seen_action_patterns.sequence_feature_accuracy=1.00
held_out_object.frequency_accuracy=0.00
held_out_object.unsupported_rate=1.00
held_out_object.state_aware_accuracy=1.00
held_out_object.transformer_ready_accuracy=0.00
held_out_object.sequence_feature_accuracy=0.00
longer_chain.frequency_accuracy=0.00
longer_chain.state_aware_accuracy=1.00
longer_chain.transformer_ready_accuracy=0.00
longer_chain.sequence_feature_accuracy=0.00
missing_link.frequency_accuracy=1.00
missing_link.state_aware_accuracy=1.00
missing_link.state_aware_unsupported_rate=1.00
missing_link.transformer_ready_accuracy=1.00
missing_link.sequence_feature_accuracy=0.00
noisy_distractor.frequency_accuracy=0.00
noisy_distractor.state_aware_accuracy=1.00
noisy_distractor.transformer_ready_accuracy=0.00
noisy_distractor.sequence_feature_accuracy=0.00
prediction_error=unsupported
update_success=True
training_size=6->7
```

## What This Shows

```text
small environment-generated data can beat a hand-written rule baseline
held-out object evaluation exposes a current frequency lookup failure
longer chain and noisy distractor slices expose more frequency lookup failures
missing link shows that unsupported can be the correct prediction
using current state relations can fix these specific held-out failures
the project now has a Transformer-ready token sequence interface
the dependency-free sequence-feature baseline fails outside seen sequences
an unsupported case can become predictable after prediction-error update
```

## What This Does Not Show

```text
latent state
trained Transformer-based prediction
learned representation updates
large data
noise
partial observation
out-of-distribution generalization
planning or control
learned state abstraction
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
1. add held-out action / delayed-effect cases
2. replace the Transformer-ready adapter and sequence-feature baseline with a trained sequence model
```
