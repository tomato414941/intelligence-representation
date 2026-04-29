# Current Results

## Role

This document records the current project conclusion. Historical experiment
details remain under `docs/legacy/`.

Use these documents for details:

- [README](../README.md): project map and common commands
- [Evaluation](evaluation.md): evaluation concepts
- [Legacy Experiment 001](legacy/experiments/experiment-001.md): retired Signal future-prediction run
- [Legacy Experiment 002](legacy/experiments/experiment-002.md): retired rendering-context investigation
- [Legacy Experiment 003](legacy/experiments/experiment-003.md): natural language held-out loss smoke check

## Current Position

The project has not produced a predictive representation system, a latent world model,
or robust action-conditioned future prediction.

The active milestone is narrower:

```text
raw examples
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
  -> task-specific output layer
```

The conceptual center remains:

```text
A predictive representation system for language, perception, action, memory, and belief.
```

World modeling is one evaluation surface inside that broader frame. It should
be evaluated through prediction of future observations or consequences, not by
next-token loss alone.

## Current Evidence

Currently supported:

```text
causal text model training utilities
byte-level and simple byte-pair text tokenization
token-level loss masks for text scoring
Fashion-MNIST image-choice raw examples
ImagePatchInputLayer -> SharedTransformerCore -> ClassificationHead
image-conditioned text candidate scoring
image-choice scoring evaluation
grid-world action-conditioned smoke data
run summary aggregation and comparison
```

Not yet supported:

```text
large-scale multimodal token learning
image-to-text continuation training
shared text/image training through one core
held-out action-conditioned future prediction improvement
latent predictive state
belief update
memory read/write learning
planning or control
```

Next-token loss reduction remains only a smoke signal. It is not evidence that
a predictive representation system or world model has been learned.

## Historical Reading

The retired Signal experiments showed that a small byte-level causal text model can reduce
loss on rendered streams, but did not show robust action-conditioned future
prediction. They also exposed that representation and rendering choices can
hide the relevant prefix from a short-context scorer.

Those results should be read as negative or diagnostic evidence for the retired
path, not as the current implementation direction.

Experiment 003 showed that the training path and small Transformer can reduce
held-out loss on a small natural-language-like corpus:

```text
initial_eval_loss: 5.6550
final_eval_loss: 2.3727
```

This supports only the narrow claim that the small training stack can learn
local text-like patterns in a smoke setting.

## Image Path

The current image path is wired end to end:

```text
Fashion-MNIST IDX
  -> image-choice JSONL
  -> local PGM files
  -> ImageChoiceExample
  -> image patch embedding
  -> SharedTransformerCore
  -> ClassificationHead
```

This is still a classification-head baseline. The next direction is to connect
image inputs and label text to continuation scoring or token-level supervision,
without reintroducing a generic raw-data envelope.

## Tokenizer Direction

The text tokenizer work should attach to raw text examples. It should not depend
on the retired Signal text path.

The current tokenizer is intentionally not production-grade. It keeps byte
fallback and supports simple byte-pair merges so tokenization can be compared
without adding external tokenizer state or dependencies.

## Next Pressure

The next useful pressure is:

```text
image-choice continuation scoring
text raw-example training path
shared core tests across image and text input layers
clear separation between tokenizer, input layer, core, and head
```

Do not add new semantic taxonomies or fixed state schemas unless a concrete
training or evaluation path forces them.
