# Current Evidence

This document records what has actually been demonstrated in the current
project. It is not a design spec, command reference, dataset guide, or complete
experiment log.

For related context:

- [README](../README.md): project map and common commands
- [Evaluation](evaluation.md): evaluation concepts
- [Current Experiment](current-experiment.md): planned or active experiment
- [Datasets](datasets.md): supported and candidate datasets
- [Legacy Docs](../legacy/docs/): retired experiments and historical notes

## Current Claim

The project has not yet produced a general predictive representation system, a
latent world model, or robust action-conditioned future prediction.

The current evidence supports a narrower claim:

```text
text, image, and image/text tasks can be connected to small Transformer-based
training paths, and those paths can reduce task losses or improve task metrics
on small to medium local runs.
```

This is useful engineering evidence. It is not yet evidence that the broader
conceptual goal has been achieved:

```text
A predictive representation system for language, perception, action, memory,
and belief.
```

## Confirmed Capabilities

The current implementation has demonstrated:

- text language-model training with held-out evaluation
- reusable byte-level BPE tokenizer training and restore
- mixed text-corpus language-model training
- image classification on MNIST, Fashion-MNIST, and CIFAR-10 subsets
- image/text choice learning with prompt text and candidate labels
- mixed text language-model and image/text choice updates in one run
- small image-conditioned text-output overfit checks
- action-conditioned grid step prediction from full grid observation and action id

## Representative Results

### Text

Tiny Shakespeare, byte tokenizer, small model:

```text
steps: 1000
eval loss: 5.7576 -> 2.3971
eval perplexity: 316.58 -> 10.99
```

Tiny Shakespeare, byte-level BPE, vocab 512, small model:

```text
steps: 1000
train tokens: 511494
eval loss: 6.4169 -> 3.9908
eval perplexity: 54.10
```

Mixed text corpus with Hugging Face byte-level BPE:

```text
corpora: Tiny Shakespeare, WikiText-2 raw train, TinyStories sample
sample size: about 17MB total
tokenizer vocab: 2048
model: small
steps: 1000
train tokens: 5047211
eval loss: 7.7803 -> 4.1420
eval perplexity: 2392.99 -> 62.93
```

Increasing the same tokenizer to vocab 8192 did not improve that 1000-step CPU
run:

```text
tokenizer vocab: 8192
model: small
steps: 1000
train tokens: 3939335
eval loss: 9.1603 -> 4.2991
eval perplexity: 9495.50 -> 73.64
```

Under this run budget, vocab 2048 remains the better baseline.

### Image Classification

Fashion-MNIST subset:

```text
train examples: 5000
eval examples: 1000
steps: 2000
train loss: 2.3432 -> 0.5871
train accuracy: 0.7888
eval accuracy: 0.7700
```

MNIST subset:

```text
train examples: 5000
eval examples: 1000
steps: 2000
train loss: 2.3495 -> 0.4515
train accuracy: 0.8574
eval accuracy: 0.8000
```

CIFAR-10 subset:

```text
train examples: 5000
eval examples: 1000
steps: 1000
train loss: 2.3366 -> 1.8463
train accuracy: 0.3176
eval accuracy: 0.3370
```

### Image/Text

Fashion-MNIST image/text choice subset with one prompt:

```text
train examples: 5000
eval examples: 1000
steps: 1000
prompt: What is this item?
train loss: 2.3271 -> 0.8679
train accuracy: 0.6926
eval accuracy: 0.6700
```

The same task with multiple prompt phrasings:

```text
steps: 1000
eval accuracy by prompt:
  What is this item?: 0.6040
  Choose the best label.: 0.6010
  Which item is shown?: 0.6180
```

Mixed text language modeling and image/text choice updates:

```text
text loss: 6.4175 -> 4.7697
image eval accuracy: 0.5630
```

Small image-conditioned text-output overfit check:

```text
examples: 2
prompt: answer:
train loss: 5.3600 -> 0.0020
generated outputs:
  image_a: A
  image_b: B
```

### World / Action

GridWorld transition table:

```text
world: 3x2 fully observed grid, 1 wall, 1 goal
train cases: 25
objective: predict next agent cell, reward class, and terminated flag
steps: 200
loss: 3.5404 -> 0.5955
final next-cell loss: 0.3412
final reward loss: 0.2488
final terminated loss: 0.0056
next-cell accuracy: 0.8000
reward accuracy: 0.8400
terminated accuracy: 1.0000
```

GridWorld held-out agent cell:

```text
world: 3x2 fully observed grid, 1 wall, 1 goal
held-out current agent cell: (0, 2)
train cases: 20
eval cases: 5
objective: predict next agent cell, reward class, and terminated flag
steps: 200
train loss: 3.4889 -> 0.1463
train next-cell accuracy: 0.9000
train reward accuracy: 1.0000
train terminated accuracy: 1.0000
eval loss: 11.3529
eval next-cell accuracy: 0.0000
eval reward accuracy: 0.6000
eval terminated accuracy: 0.8000
```

## What These Results Mean

The results show that the training paths are functional and that small models
can learn measurable patterns from text, image, and image/text data.
The GridWorld held-out-cell run also shows a current limitation: the small
grid step predictor can fit seen transition cases, but it has not yet shown
reliable state generalization.

They do not show:

- large-scale multimodal generation
- robust image-conditioned open-ended text generation
- latent predictive state learning
- belief update or memory read/write learning
- planning or control
- action-conditioned future prediction that generalizes beyond the small grid
  transition table

Next-token loss reduction, classification accuracy, and choice accuracy are
important smoke and task metrics. They are not sufficient by themselves to
claim that a world model or general predictive representation system has been
learned.

## Historical Reading

Retired Signal-era experiments showed that a small byte-level causal text model
could reduce loss on rendered streams, but did not show robust
action-conditioned future prediction. Those runs are diagnostic history, not
the current implementation direction.

Legacy experiment details remain under `legacy/docs/`.
