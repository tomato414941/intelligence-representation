# Current Results

## Position

This document records the current prototype result after retiring the old experiment tree.

The project has not produced a predictive token machine or a latent world model. The main direction has shifted from the old symbolic prediction benchmark toward a typed token stream prediction scaffold:

```text
A predictive token machine for language, perception, action, memory, and belief.
```

World modeling remains a core evaluation surface inside that broader frame: it asks whether observation/action history can predict future observations or consequences.

The new v1 foundation now separates the legacy smoke corpus from the active typed stream path. Legacy `MixedDocument` remains for compatibility, while new PTM-oriented experiments should use `TypedEvent` JSONL and `FuturePredictionCase` ranking.

The typed stream path can represent:

```text
text
observations
actions
consequences
tool calls and tool results
prediction and prediction_error events
state / belief / memory / reward roles as typed event envelopes
```

This is not an OpenAI API wrapper and not a pretrained chat model. It uses the GPT/Transformer sequence-learning pattern directly.

The current training objective is next-token prediction, but the project should not treat next-token loss reduction as evidence of a learned predictive token machine or world model. World-model-oriented evidence must come from target-position-aware future prediction, especially held-out consequence ranking after observation/action prefixes.

The built-in corpus is the smoke corpus only. It is deliberately small and exists for demos, tests, and quick loss-reduction checks. The main corpus growth path is JSONL files loaded through the training CLI, where larger project-owned and public/internet-sourced mixed data can be added without expanding code-level schemas or taxonomies.

Historical experiment code and notes live under:

- [legacy/experiments/](../legacy/experiments/)
- [legacy/tests/](../legacy/tests/)
- [docs/legacy/](legacy/)

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

intrep.grid_world / intrep.grid_corpus:
  small hidden-state grid transitions rendered as text/grid/action/next-grid documents

intrep.gpt_model / intrep.gpt_training / intrep.train_gpt:
  decoder-only GPT, next-token training loop, and CLI entrypoint

intrep.pair_ranking:
  symbolic-to-natural environment pair ranking by continuation loss

intrep.next_observation_cases / intrep.next_observation_ranking / intrep.next_observation_evaluation:
  mixed observation-plus-action to next-observation ranking before and after GPT training,
  with same-modality hard distractors by default

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
corpus=builtin tokens=1990 steps=20 initial_loss=... final_loss=...
```

The important condition is not high capability. It is that an untrained decoder-only GPT can consume the built-in smoke corpus and reduce next-token loss in a short run.

The corresponding real-data path is:

```sh
uv run python -m intrep.train_gpt --corpus file --corpus-path path/to/corpus.jsonl --loss-summary
```

JSONL file corpora are the intended place for the main corpus to grow.

Public or internet-sourced data should use the same file-backed corpus path after external collection, filtering, licensing/provenance review, and train/eval splitting. That keeps data growth on the main GPT corpus line without adding a Semantic State layer or a crawler-specific architecture.

A small non-language observation smoke corpus is also available:

```sh
uv run python -m intrep.train_gpt --corpus builtin-grid --max-steps 5 --loss-summary
```

This keeps the same byte-level GPT training path, but adds grid observations and action-conditioned next observations. It is still a toy corpus, not a true multimodal model.

The first correspondence metric is available as a library-level evaluator:

```text
symbolic environment text as prefix
candidate natural-language episode descriptions as continuations
rank by continuation loss
```

This is a diagnostic for whether the model assigns lower loss to the paired natural-language description. It is not a capability claim.

The corresponding before/after training evaluator supports held-out environment pairs:

```text
train documents = mixed corpus without selected environment pairs
eval documents = held-out symbolic/natural environment pairs
report untrained vs trained symbolic-to-natural ranking metrics
```

A second diagnostic extracts mixed next-observation cases:

```text
prefix = observation + action
positive continuation = correct next observation
distractors = same-modality next observations from other cases by default
rank by continuation loss
```

This covers symbolic environment text and grid observations through the same evaluation shape, so grid does not become the only target. Use `--distractor-policy all_other` to reproduce the earlier cross-modality distractor pool.

The independent evaluation runner compares untrained and trained ranking scores:

```sh
uv run python -m intrep.evaluate_next_observation --max-steps 5
```

For held-out ranking, pass a separate eval JSONL corpus:

```sh
uv run python -m intrep.evaluate_next_observation \
  --corpus file --corpus-path train.jsonl \
  --eval-corpus-path eval.jsonl
```

The generated environment split can be selected directly without first writing JSONL:

```sh
uv run python -m intrep.evaluate_next_observation \
  --corpus generated-environment \
  --generated-eval-slice generated_held_out_object
```

If no eval corpus is provided, the runner reports training-set ranking only. That is a smoke check, not a generalization claim.

## RunPod Short Sweep

A short RunPod sweep was run on `main` at commit `b2d8e03` with CUDA available, seed `7`, `max_steps=40`, and `distractor_policy=same_entity`.

The generated environment slices showed next-token loss reduction:

```text
generated_seen:
  final_eval_loss: 5.714 -> 2.870
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083
  eval_cases: 12

generated_held_out_object:
  final_eval_loss: 5.733 -> 3.252
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083
  eval_cases: 12

generated_held_out_container:
  final_eval_loss: 5.716 -> 3.335
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083
  eval_cases: 12

generated_held_out_location:
  final_eval_loss: 5.728 -> 3.468
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.167 -> 0.167
  eval_cases: 6
```

This supports only the narrow statement that the small byte-level GPT learned to reduce continuation loss on the generated text streams. It does not show improved action-conditioned next-observation prediction.

In this historical RunPod run, the `generated_strict_*` slices were not yet usable as ranking metrics because each slice had only one eval case. The current code expands each strict slice to four symbolic eval cases, so the next step is to rerun the generated-environment sweep and inspect strict ranking metrics instead of treating them as skipped.

## Support Benchmark

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
general predictive token machine behavior
latent state
large or strong Transformer-based prediction
learned representation updates
large data
noise
partial observation
out-of-distribution generalization
planning or control
learned state abstraction
improved action-conditioned next-observation ranking
```

The current milestone is not "a predictive token machine is built" or "a world model is built."

It is:

```text
the repository now has a small typed token stream GPT scaffold
```

## Next Pressure

Next work should not add new taxonomies or semantic state schemas.

It should either:

```text
1. rerun generated-environment strict slices now that ranking metrics run
2. expand the mixed corpus while keeping the same decoder-only GPT training path
3. add simple held-out evaluation for environment-text correspondences
4. keep symbolic predictor work as regression/support only
```
