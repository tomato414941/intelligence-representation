# Experiment 003: Natural Language Learning Smoke

## Purpose

This experiment checks whether the current small decoder-only Transformer can
learn ordinary natural-language-like local patterns before interpreting harder
world-modeling failures.

The question is intentionally narrow:

```text
Can the current byte-level GPT reduce held-out next-token loss on a small
natural language corpus?
```

This is not a Predictive Token Machine capability claim. It is a sanity check
for the training path.

## Setup

The corpus is a small project-owned toy natural language corpus.

```text
train documents: 80
eval documents: 24
train byte tokens: 11839
eval byte tokens: 3605
```

The records use the legacy mixed-document JSONL shape:

```text
id
modality = text
content
```

Generated files:

```text
runs/exp003_natural_language/train.jsonl
runs/exp003_natural_language/eval.jsonl
runs/exp003_natural_language/metrics_steps100.json
```

## Command

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m intrep.train_signal_text \
  --train-path runs/exp003_natural_language/train.jsonl \
  --eval-path runs/exp003_natural_language/eval.jsonl \
  --max-steps 100 \
  --context-length 64 \
  --batch-size 8 \
  --batch-stride 32 \
  --loss-history-path runs/exp003_natural_language/metrics_steps100.json
```

Configuration:

```text
model_preset = small
embedding_dim = 32
num_heads = 4
hidden_dim = 64
num_layers = 1
context_length = 64
batch_size = 8
batch_stride = 32
max_steps = 100
device = cpu
eval_split = held_out
```

## Results

```text
initial_step_loss: 5.7066
final_step_loss: 2.2480
best_step_loss: 2.0496
loss_reduction: 3.4586
loss_reduction_ratio: 60.61%

initial_train_loss: 5.7031
final_train_loss: 2.1212

initial_eval_loss: 5.6550
final_eval_loss: 2.3727
```

## Reading

The current small Transformer can learn local natural-language-like patterns in
this toy corpus. The held-out eval loss decreased substantially:

```text
5.6550 -> 2.3727
```

This means the hard-negative world-stream ranking failures should not be read as
"the model cannot learn anything." They should be investigated as more specific
failures involving:

```text
hard-negative task structure
rendering
context length
ranking scorer
training budget
model capacity
data distribution symmetry
```

## Non-Claims

This experiment does not show:

```text
general language modeling ability
semantic understanding
world-modeling ability
Predictive Token Machine capability
robust held-out generalization beyond the toy distribution
```

It only establishes that the current training path and small Transformer are
capable of reducing held-out loss on a simple natural-language-like corpus.
