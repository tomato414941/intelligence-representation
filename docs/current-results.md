# Current Results

## Role

This document records the current project conclusion and points to the detailed
experiment notes. It is not the implementation map, command reference, or full
run log.

Use these documents for details:

- [README](../README.md): project map and common commands
- [Evaluation](evaluation.md): evaluation concepts and CLI shape
- [Experiment 001](experiment-001.md): first hard-negative Signal future-prediction run
- [Experiment 002](experiment-002.md): 100x data and rendering-context investigation
- [Experiment 003](experiment-003.md): natural language held-out loss smoke check
- [RunPod Training Notes](runpod.md): GPU execution notes

## Current Position

The project has not produced a predictive token machine, a latent world model,
or robust action-conditioned future prediction.

The current milestone is narrower:

```text
a small Signal stream GPT scaffold with target-position future-prediction evaluation
```

The conceptual center remains:

```text
A predictive token machine for language, perception, action, memory, and belief.
```

World modeling is one evaluation surface inside that broader frame. It asks
whether observation/action history can predict future observations or
consequences.

## Current Evidence

What is currently supported:

```text
Signal streams can represent observations, actions, consequences, tool
events, prediction errors, and belief/memory-like records.

A small decoder-only GPT can reduce next-token loss on the available smoke and
generated text streams.

The same training path can reduce held-out loss on a small natural-language-like
toy corpus.

Generated hard-negative signal environment data can be produced at 100x the
Experiment 001 scale.

FuturePredictionCase ranking can target consequence events with same-channel
hard-negative distractors.
```

What is not yet supported:

```text
held-out action-conditioned ranking improvement
robust context-conditioned consequence discrimination
latent predictive state
belief update
memory read/write learning
large-scale multimodal token learning
planning or control
```

Next-token loss reduction remains only a smoke signal. It is not evidence that
a predictive token machine or world model has been learned.

## Experiment Reading

### Experiment 001

Experiment 001 generated two hard-negative signal environment slices:

```text
same_history_different_action
same_action_different_context
```

The short run reduced next-token loss but did not improve hard-negative
future-prediction ranking.

Original reading:

```text
The byte-level signal-tag GPT scaffold learned local stream statistics, but did
not show action/context-conditioned consequence discrimination.
```

Updated caveat:

```text
The run used full signal rendering. With context_length = 64, the rendered
tags were long enough that consequence payload could be scored after the
observation/action prefix had fallen out of the model window.
```

So Experiment 001 remains a negative measured result, but it is not a clean
modeling failure. It also exposed an evaluation rendering problem.

### Experiment 002

Experiment 002 scaled the generated hard-negative Signal data to:

```text
per condition:
  train_cases = 8000
  eval_cases = 3200
  explicit_negative_rate = 0.0
```

It also separated future-prediction ranking rendering into:

```text
signal:
  full Signal tag rendering, treated as a low-priority experiment

payload:
  rendered event payloads only, better for the current short context length
  in the text/byte-tokenizer evaluation path
```

The key finding:

```text
signal rendering can hide the relevant observation/action prefix from the
scorer under context_length = 64.
```

Small 100x probes with `rendering=payload` removed the exact zero-margin
artifact, but did not produce top-1 ranking improvement:

```text
same_history_different_action:
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: -0.0025
  after_margin: 0.0013

same_action_different_context:
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: -0.0032
  after_margin: -0.0246
```

Current reading:

```text
Data scale alone has not produced a clean ranking improvement in the small CPU
probe.

Payload rendering is the better diagnostic for the current context length.

The next experiments should keep the hard-negative evaluation fixed and vary
only the factors that affect whether the model can see and use the relevant
prefix.
```

### Experiment 003

Experiment 003 tested the current byte-level small GPT on a toy natural language
corpus:

```text
train documents = 80
eval documents = 24
train byte tokens = 11839
eval byte tokens = 3605
max_steps = 100
```

Held-out eval loss decreased substantially:

```text
initial_eval_loss: 5.6550
final_eval_loss: 2.3727
```

Current reading:

```text
The training path and small Transformer can learn local natural-language-like
patterns. Hard-negative world-stream ranking failures should therefore be
investigated as task/rendering/context/scoring/model-capacity issues, not as
evidence that the model cannot learn at all.
```

## Historical RunPod Sweep

A historical RunPod sweep on `main` at commit `b2d8e03` used CUDA, seed `7`,
`max_steps = 40`, and `distractor_policy = same_entity`.

The generated environment slices showed next-token loss reduction:

```text
generated_seen:
  final_eval_loss: 5.714 -> 2.870
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083

generated_held_out_object:
  final_eval_loss: 5.733 -> 3.252
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083

generated_held_out_container:
  final_eval_loss: 5.716 -> 3.335
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.083 -> 0.083

generated_held_out_location:
  final_eval_loss: 5.728 -> 3.468
  next_observation_accuracy: 0.000 -> 0.000
  symbolic_to_natural_accuracy: 0.167 -> 0.167
```

This historical run supports only the narrow statement that the small byte-level
GPT reduced continuation loss on generated text streams. It did not show
improved action-conditioned next-observation prediction.

## Support Benchmark

The old symbolic benchmark remains useful as a regression and contrast, but it
is not the main project path.

It shows:

```text
rule baselines are brittle
frequency predictors memorize seen transitions
state-aware predictors can fix specific held-out symbolic cases
sequence-feature and tiny Transformer baselines expose current generalization limits
prediction-error update can make one unsupported case predictable
```

Do not read the symbolic benchmark as evidence for the PTM hypothesis. It is a
support surface for regression and contrast.

## Next Pressure

Next work should not add new semantic taxonomies or fixed state schemas.

The next useful pressure is:

```text
rendering = payload
context_length
max_steps
model size
train-set versus held-out ranking
eval case count once scoring cost is acceptable
```

## Fashion-MNIST Image Signals

The first image-signal smoke path is now wired end to end:

```text
Fashion-MNIST IDX
  -> local PGM files
  -> image Signal payload_ref
  -> patch-token rendering
  -> decoder-only GPT
  -> image-to-label future prediction ranking
```

The task-specific entry point is `intrep.evaluate_fashion_mnist`; the generic
future-prediction CLI remains focused on text-rendered Signal streams.

Initial CPU smoke result with `patch_size = 4`, `channel_bins = 4`,
`model_preset = tiny`, `train_cases = 20`, `max_steps = 100`, and
`max_negatives = 3`:

```text
train split:
  before_top1_accuracy = 0.1500
  after_top1_accuracy = 0.3500
  delta_top1_accuracy = 0.2000
  before_margin = -0.0505
  after_margin = -0.3269
```

This is a weak train-split memorization signal, not evidence of image
generalization. A held-out 100/20 smoke with the same tiny setup worsened
top-1 accuracy from `0.2500` to `0.1000`. The current ranking implementation is
also slow unless `max_negatives` is used, so broader Fashion-MNIST runs should
first improve evaluation throughput or use smaller diagnostic subsets.

The immediate experimental question is:

```text
Can the small decoder-only predictor improve hard-negative consequence ranking
when the relevant observation/action prefix is actually visible to the scorer?
```

Only after that should the project consider broader tokenizer, corpus, memory,
or architecture changes.
