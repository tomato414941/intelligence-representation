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
  --train-size 100 \
  --eval-size 40 \
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
  --train-size 100 \
  --eval-size 40 \
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
before_margin
after_margin
train_final_loss
```

The current report is enough for the first run. A later metrics PR should add:

```text
delta_top1_accuracy
delta_margin
explicit_negative_rate
no_negative_case_count
case_count_by_condition
```

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
