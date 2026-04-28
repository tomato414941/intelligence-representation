# Experiment 001: Typed Environment Future Prediction

## Purpose

This experiment is the first mainline check for the Predictive Token Machine scaffold.

It does not try to prove that a world model has been learned. It asks a narrower question:

```text
Can a small decoder-only GPT trained on typed environment streams improve
held-out consequence ranking when the distractors require using either
the action token or the observation context?
```

The training objective remains next-token prediction. The evaluation target is future prediction at the consequence position.

## Fixed Setup

Use the generated typed environment corpus with explicit hard negatives:

```text
same_history_different_action:
  same observation
  different action
  different consequence

same_action_different_context:
  same action
  different observation
  different consequence
```

Both slices attach reciprocal `negative_event_ids` to consequence events. This makes the ranking task depend on intentional hard negatives instead of incidental same-modality distractors.

## Commands

Build the action-conditioned slice:

```sh
uv run python -m intrep.generated_environment_typed_corpus \
  --train-output runs/exp001/same_history_train.typed.jsonl \
  --eval-output runs/exp001/same_history_eval.typed.jsonl \
  --eval-slice same_history_different_action \
  --train-size 80 \
  --eval-size 32 \
  --seed 7
```

Evaluate it:

```sh
uv run python -m intrep.evaluate_future_prediction \
  --train-path runs/exp001/same_history_train.typed.jsonl \
  --eval-path runs/exp001/same_history_eval.typed.jsonl \
  --target-role consequence \
  --condition same_history_different_action \
  --metrics-path runs/exp001/same_history_metrics.json
```

Build the context-conditioned slice:

```sh
uv run python -m intrep.generated_environment_typed_corpus \
  --train-output runs/exp001/same_action_train.typed.jsonl \
  --eval-output runs/exp001/same_action_eval.typed.jsonl \
  --eval-slice same_action_different_context \
  --train-size 80 \
  --eval-size 32 \
  --seed 7
```

Evaluate it:

```sh
uv run python -m intrep.evaluate_future_prediction \
  --train-path runs/exp001/same_action_train.typed.jsonl \
  --eval-path runs/exp001/same_action_eval.typed.jsonl \
  --target-role consequence \
  --condition same_action_different_context \
  --metrics-path runs/exp001/same_action_metrics.json
```

## Primary Metrics

Record these for each slice:

```text
train_case_count
eval_case_count
generalization_eval
before_top1_accuracy
after_top1_accuracy
delta_top1_accuracy
before_margin
after_margin
delta_margin
explicit_negative_rate
no_negative_case_count
train_final_loss
```

The metrics report should make clear whether the ranking task actually used explicit hard negatives. `explicit_negative_rate` should be `1.0` for the generated hard-negative slices.

## Success Criteria

The experiment is successful as an engineering checkpoint if:

```text
1. train/eval typed JSONL files are generated.
2. eval_case_count >= 2 for both slices.
3. generalization_eval = true.
4. each eval consequence has at least one explicit negative.
5. before/after future prediction metrics are emitted.
6. all repository tests still pass.
```

It is successful as a modeling signal only if:

```text
after_top1_accuracy > before_top1_accuracy
and
after_margin > before_margin
on held-out eval cases.
```

Loss reduction alone is not a modeling success.

## How To Read Results

If train loss decreases but ranking does not improve:

```text
The model learned local sequence statistics or formatting, but not enough
action-conditioned future prediction to affect this evaluation.
```

If ranking improves only without an eval path:

```text
This is train-set smoke behavior. Do not interpret it as generalization.
```

If held-out ranking improves on `same_history_different_action`:

```text
The model is using the action token enough to prefer the correct consequence
over an explicit action-contrast negative.
```

If held-out ranking improves on `same_action_different_context`:

```text
The model is using the observation context enough to prefer the correct
consequence over an explicit context-contrast negative.
```

If both improve:

```text
This is the first useful evidence that the typed stream scaffold can train
toward action/context-conditioned future prediction on the generated task.
It is still not evidence of a general predictive token machine.
```

## Non-Goals

This experiment does not test:

```text
role embeddings
special tokenizers
vision/audio streams
belief or memory usefulness
planning/control
large-scale generalization
pretrained language model transfer
```

Those should wait until the typed corpus and future-prediction metrics are stable.

## First Run

Date: 2026-04-27

Configuration:

```text
train_size = 80
eval_size = 32
seed = 7
target_role = consequence
eval_split = held_out
generalization_eval = true
```

Results:

```text
same_history_different_action:
  train_case_count: 80
  eval_case_count: 32
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: -0.00000016
  after_margin: -0.00000017
  train_final_loss: 3.5707

same_action_different_context:
  train_case_count: 80
  eval_case_count: 32
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  before_margin: 0.0000
  after_margin: 0.0000
  train_final_loss: 3.4959
```

Additional train-split smoke check:

```text
same_history_different_action:
  eval_split: train
  generalization_eval: false
  eval_case_count: 80
  explicit_negative_rate: 1.0
  no_negative_case_count: 0
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  delta_top1_accuracy: 0.0000
  before_margin: -0.00000015
  after_margin: -0.00000004
  delta_margin: 0.00000011

same_action_different_context:
  eval_split: train
  generalization_eval: false
  eval_case_count: 80
  explicit_negative_rate: 1.0
  no_negative_case_count: 0
  before_top1_accuracy: 0.5000
  after_top1_accuracy: 0.5000
  delta_top1_accuracy: 0.0000
  before_margin: 0.0000
  after_margin: 0.0000
  delta_margin: 0.0000
```

Reading:

```text
The short training run reduced next-token loss, but it did not improve
held-out hard-negative future prediction ranking.
```

This is a useful negative result. It means the current byte-level typed-tag GPT scaffold can learn local stream statistics on this corpus, but this first run does not show action-conditioned or context-conditioned consequence discrimination.

The train-split smoke check sharpens the reading: the failure is not only held-out generalization. Under the short default training run, the model does not distinguish the explicit hard negatives even on the training split.

Next steps should not claim world-model evidence from this run. The next experimental step is to vary training steps, model size, context length, and tokenizer/rendering while keeping this hard-negative evaluation fixed.

## Postscript: Rendering Caveat

A follow-up investigation found that this first run should be read with an
important evaluation caveat.

The run used full typed-event rendering for ranking prefixes and continuations.
With `context_length = 64`, the rendered `<EVENT ...>` tags were long enough
that the consequence content was often scored after the observation/action
prefix had fallen out of the model window.

For example, in a `same_action_different_context` case:

```text
typed-event prefix: 844 byte tokens
typed-event positive continuation: 523 byte tokens
content-only prefix: 27 byte tokens
content-only positive continuation: 9 byte tokens
```

This means the zero or near-zero margins in the first run are not pure evidence
that the model could not use action/context information. They also reflect an
evaluation rendering problem: the ranking scorer could be comparing consequence
continuations without the relevant causal prefix still in context.

The first-run negative result should still be preserved:

```text
With full typed-event rendering and a short training run, the measured ranking
did not improve.
```

But it should not be over-interpreted as a clean modeling failure. The follow-up
experiment records the 100x data generation, scoring-speed change, and
content-rendering probe:

- [Experiment 002](experiment-002.md)
