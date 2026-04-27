# Current Results

## Position

This document records the current prototype result after retiring the old experiment tree.

The project has not produced a latent world model. The main direction has shifted from the old symbolic prediction benchmark toward a small mixed-world decoder-only GPT training foundation.

The new v1 foundation trains an untrained GPT-style model on a mixed corpus:

```text
natural language
environment episodes in symbolic form
environment episodes in natural-language form
code
logs / tool-like outputs
```

This is not an OpenAI API wrapper and not a pretrained chat model. It uses the GPT/Transformer sequence-learning pattern directly.

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

intrep.torch_sequence / intrep.tiny_transformer:
  vocabulary, tensors, and a tiny trained Transformer predictor

intrep.byte_tokenizer:
  byte-level tokenizer for mixed Unicode/code/log text

intrep.mixed_corpus:
  minimal mixed-world corpus samples

intrep.gpt_model / intrep.gpt_training / intrep.train_gpt:
  decoder-only GPT, next-token training loop, and CLI entrypoint

intrep.evaluation:
  evaluate_prediction_cases

intrep.benchmark:
  run_benchmark

intrep.update_loop:
  PredictionErrorUpdateLoop
```

## Current Main Training Check

The canonical new training smoke check is:

```sh
uv run python -m intrep.train_gpt --max-steps 20
```

Expected shape:

```text
intrep mixed-gpt training
corpus=builtin tokens=1025 steps=20 initial_loss=... final_loss=...
```

The important condition is not high capability. It is that an untrained decoder-only GPT can consume the mixed corpus and reduce next-token loss in a short run.

## Current Benchmark

The old symbolic benchmark is still executable as `intrep.benchmark.run_benchmark()`.

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

tiny Transformer:
  trains a small TransformerEncoder on the same token sequences

condition slices:
  expose seen patterns, held-out objects, longer chains, missing links, and noisy distractors

generated distribution:
  evaluates repeated find patterns across generated object/container/location combinations

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
tiny_transformer_accuracy=0.40
seen_action_patterns.frequency_accuracy=1.00
seen_action_patterns.state_aware_accuracy=1.00
seen_action_patterns.transformer_ready_accuracy=1.00
seen_action_patterns.sequence_feature_accuracy=1.00
seen_action_patterns.tiny_transformer_accuracy=1.00
held_out_object.frequency_accuracy=0.00
held_out_object.unsupported_rate=1.00
held_out_object.state_aware_accuracy=1.00
held_out_object.transformer_ready_accuracy=0.00
held_out_object.sequence_feature_accuracy=0.00
held_out_object.tiny_transformer_accuracy=0.00
longer_chain.frequency_accuracy=0.00
longer_chain.state_aware_accuracy=1.00
longer_chain.transformer_ready_accuracy=0.00
longer_chain.sequence_feature_accuracy=0.00
longer_chain.tiny_transformer_accuracy=0.00
missing_link.frequency_accuracy=1.00
missing_link.state_aware_accuracy=1.00
missing_link.state_aware_unsupported_rate=1.00
missing_link.transformer_ready_accuracy=1.00
missing_link.sequence_feature_accuracy=0.00
missing_link.tiny_transformer_accuracy=0.00
noisy_distractor.frequency_accuracy=0.00
noisy_distractor.state_aware_accuracy=1.00
noisy_distractor.transformer_ready_accuracy=0.00
noisy_distractor.sequence_feature_accuracy=0.00
noisy_distractor.tiny_transformer_accuracy=0.00
generated_train_size=12
generated_seen.tiny_transformer_accuracy=1.00
generated_held_out_object.tiny_transformer_accuracy=0.00
generated_held_out_container.tiny_transformer_accuracy=0.50
generated_held_out_location.tiny_transformer_accuracy=0.00
prediction_error=unsupported
update_success=True
training_size=6->7
```

## What The Symbolic Benchmark Shows

```text
small environment-generated data can beat a hand-written rule baseline
held-out object evaluation exposes a current frequency lookup failure
longer chain and noisy distractor slices expose more frequency lookup failures
missing link shows that unsupported can be the correct prediction
using current state relations can fix these specific held-out failures
the project now has a Transformer-ready token sequence interface
the dependency-free sequence-feature baseline fails outside seen sequences
a tiny trained Transformer is now on the same benchmark
generated distribution shows the tiny Transformer still mostly memorizes seen combinations
an unsupported case can become predictable after prediction-error update
```

## What The Current Project Still Does Not Show

```text
latent state
large or strong Transformer-based prediction
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
the repository now has a small mixed-world GPT training foundation
```

## Next Pressure

Next work should not add new taxonomies or semantic state schemas.

It should either:

```text
1. expand the mixed corpus while keeping the same decoder-only GPT training path
2. add simple held-out evaluation for environment-text correspondences
3. keep symbolic predictor work as regression/support only
```
