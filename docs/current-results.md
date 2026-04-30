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
byte-pair text tokenization as the main text path
byte-level text tokenization as a small fallback baseline
Tiny Shakespeare language modeling run with held-out evaluation
Tiny Shakespeare byte-level BPE checkpoint with tokenizer restore
token-level loss masks for text scoring
Fashion-MNIST image-choice raw examples
ImagePatchInputLayer -> SharedTransformerCore -> ClassificationHead
image-conditioned text candidate scoring
image-choice scoring evaluation
grid-world action-conditioned smoke data
```

Not yet supported:

```text
large-scale multimodal token learning
large-scale image-to-text continuation training
held-out action-conditioned future prediction improvement
latent predictive state
belief update
memory read/write learning
planning or control
```

Tiny Shakespeare was trained as a real text corpus run, not a toy fixture:

```text
corpus: Tiny Shakespeare
model: small causal text model
tokenizer: byte
train/eval split: 90/10
steps: 1000
eval loss: 5.7576 -> 2.3971
eval perplexity: 316.58 -> 10.99
```

The first byte-level BPE run restores its tokenizer from the checkpoint and can
generate text without falling back to the byte tokenizer:

```text
corpus: Tiny Shakespeare
model: small causal text model
tokenizer: byte-level BPE, vocab 512
train/eval split: 90/10
steps: 1000
train tokens: 511494
eval loss: 6.4169 -> 3.9908
eval perplexity: 54.10
```

Mixed text corpus training now accepts multiple `--corpus-path` values. Each
corpus is split into train/eval before the splits are shuffled and joined with
an end-of-text separator:

```text
corpora: Tiny Shakespeare sample, WikiText-2 sample, TinyStories sample
model: tiny causal text model
tokenizer: byte-level BPE, vocab 512
train/eval split: per corpus, 90/10
steps: 20
train tokens: 26087
eval loss: 6.4250 -> 6.2498
```

This is an integration check, not a quality result. Larger mixed text runs
exposed that the older eager window materialization and naive simple byte-pair
training were too slow for larger corpora on CPU.

The text language-model training loop now uses a PyTorch `Dataset` and
`DataLoader` for token windows instead of materializing every window into a
large eager tensor before training. This keeps the public behavior the same
while making larger corpus runs practical:

```text
corpora: Tiny Shakespeare sample, WikiText-2 sample, TinyStories sample
sample size: about 200KB per corpus
model: small causal text model
tokenizer: byte
steps: 100
train tokens: 540044
eval loss: 5.7667 -> 2.7941
```

```text
corpora: Tiny Shakespeare sample, WikiText-2 sample, TinyStories sample
sample size: about 200KB per corpus
model: small causal text model
tokenizer: byte-level BPE, vocab 512
steps: 100
train tokens: 268211
eval loss: 6.4252 -> 5.1240
```

Text tokenizers can now be trained and saved separately from language-model
training. The language-model CLI can reuse a saved tokenizer, avoiding repeated
BPE fitting during experiment runs:

```text
tokenizer: saved byte-level BPE, vocab 512
corpora: Tiny Shakespeare sample, WikiText-2 sample, TinyStories sample
sample size: about 200KB per corpus
model: small causal text model
steps: 100
train tokens: 267036
eval loss: 6.4150 -> 5.1430
```

The same path has been run on a larger mixed text corpus using a fixed saved
byte-level BPE tokenizer:

```text
tokenizer: saved byte-level BPE, vocab 512
corpora: Tiny Shakespeare, WikiText-2 sample, TinyStories sample
sample size: Tiny Shakespeare full text, about 1MB each for WikiText-2 and TinyStories
model: tiny causal text model
steps: 1000
train tokens: 1402456
eval loss: 6.3833 -> 4.4446
```

The main byte-pair tokenizer path now uses Hugging Face `tokenizers` byte-level
BPE. This keeps tokenizer training out of the project's custom Python code and
removes the previous naive simple-BPE bottleneck:

```text
tokenizer: Hugging Face byte-level BPE, vocab 512
corpora: Tiny Shakespeare, WikiText-2 raw train, TinyStories sample
sample size: about 17MB total
tokenizer training time: about 9 seconds on CPU
model: tiny causal text model
steps: 100
train tokens: 7490793
eval loss: 6.4698 -> 5.3470
```

The same mixed text path improves when the model preset is increased from
`tiny` to `small`:

```text
tokenizer: byte-pair, vocab 2048
corpora: Tiny Shakespeare, WikiText-2 raw train, TinyStories sample
sample size: about 17MB total
model: small causal text model
steps: 1000
train tokens: 5047211
eval loss: 7.7803 -> 4.1420
eval perplexity: 2392.99 -> 62.93
```

Increasing the same main byte-pair tokenizer to vocab 8192 works, but does not
improve the 1000-step CPU run yet:

```text
tokenizer: byte-pair, vocab 8192
corpora: Tiny Shakespeare, WikiText-2 raw train, TinyStories sample
sample size: about 17MB total
model: small causal text model
steps: 1000
train tokens: 3939335
eval loss: 9.1603 -> 4.2991
eval perplexity: 9495.50 -> 73.64
```

The larger vocabulary reduces token count, but it also increases output-layer
cost and starts from a harder softmax. Under this run budget, vocab 2048 remains
the better baseline.

Next-token loss reduction is evidence for language-model training, but it is
not evidence by itself that a predictive representation system or world model
has been learned.

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

## Image Paths

The current Fashion-MNIST image classification path is wired end to end:

```text
IDX image dataset
  -> image-choice JSONL
  -> local PGM files
  -> ImageChoiceExample
  -> image patch embedding
  -> SharedTransformerCore
  -> ClassificationHead
```

The current tiny preset learns a 5,000-example Fashion-MNIST subset on CPU:

```text
train_examples: 5000
eval_examples: 1000
max_steps: 2000
train_initial_loss: 2.3432
train_final_loss: 0.5871
train_accuracy: 0.7888
eval_accuracy: 0.7700
```

The same image classification path also learns a 5,000-example MNIST subset on
CPU by using digit label choices:

```text
train_examples: 5000
eval_examples: 1000
max_steps: 2000
train_initial_loss: 2.3495
train_final_loss: 0.4515
train_accuracy: 0.8574
eval_accuracy: 0.8000
```

CIFAR-10 can also be converted into the same image-choice JSONL shape from local
python batch files:

```text
CIFAR-10 python batch
  -> image-choice JSONL
  -> local PPM files
  -> ImageChoiceExample
  -> image patch embedding
  -> SharedTransformerCore
  -> ClassificationHead
```

The same tiny preset can learn a 5,000-example CIFAR-10 subset on CPU. CIFAR-10
is harder than MNIST-style grayscale datasets, but the RGB image path learns
above chance in a short run:

```text
train_examples: 5000
eval_examples: 1000
max_steps: 1000
train_initial_loss: 2.3366
train_final_loss: 1.8463
train_accuracy: 0.3176
eval_accuracy: 0.3370
```

The image-conditioned text scoring path is separate:

```text
ImageChoiceExample
  -> image patch embedding + candidate text embeddings
  -> SharedTransformerCore
  -> candidate continuation loss
```

The image-to-text label output path trains token loss directly against the
answer text instead of using a classification head:

```text
ImageChoiceExample
  -> image patch embedding + answer text token embeddings
  -> SharedTransformerCore
  -> token output loss on answer text tokens
```

The tiny preset learns a 5,000-example Fashion-MNIST image-to-text subset on
CPU:

```text
train_examples: 5000
eval_examples: 1000
max_steps: 1000
train_initial_loss: 5.6635
train_final_loss: 0.1884
eval_final_loss: 0.1863
```

The classification path is still the simplest image baseline. The image-to-text
path is now the first direct bridge from image inputs to token outputs, without
reintroducing a generic raw-data envelope.

## Tokenizer Direction

The text tokenizer work should attach to raw text examples. It should not depend
on the retired Signal text path.

The main byte-pair tokenizer uses the external `tokenizers` library. The
simple byte-pair tokenizer remains only as a small internal baseline.

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
