# intelligence-representation

This repository explores representation for intelligence through a small,
testable research prototype.

The conceptual center is:

```text
A predictive representation system for language, perception, action, memory, and belief.
```

The project does not aim to build a hand-designed semantic database. The current
direction is to keep source examples close to their original form, convert them
through task-appropriate input layers, and connect them to shared predictive
computation where useful.

```text
raw examples
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
  -> task-specific output layer
```

World modeling is one evaluation surface inside this broader frame. It concerns
whether observation and action history can improve predictions about future
observations, outcomes, or state changes.

Loss reduction is a training smoke signal, not evidence by itself that a
predictive representation system or world model has been learned.

## Project Map

```text
src/intrep/
  Active prototype package

tests/
  Default test suite

docs/
  Project concepts, evaluation principles, and current results

docs/current-experiment.md
  Planned or active experiment note

legacy/
  Historical docs, experiments, and tests
```

## Canonical Docs

Read these first:

- [Concept](docs/concept.md)
- [Predictive Representation System](docs/predictive-representation-system.md)
- [Model Boundaries](docs/model-boundaries.md)
- [Learning and Execution](docs/learning-and-execution.md)
- [Worlds and Experience](docs/worlds-and-experience.md)
- [Datasets](docs/datasets.md)
- [Current Experiment](docs/current-experiment.md)
- [World Model Centering](docs/world-model.md)
- [Bitter Lesson Correction](docs/bitter-lesson.md)
- [Evaluation](docs/evaluation.md)
- [Current Evidence](docs/current-evidence.md)

Historical notes:

- [Legacy Docs](legacy/docs/)

## Design Constraints

Prefer:

```text
raw examples before premature schemas
task-specific input layers
shared predictive computation where it is actually useful
loss curves as smoke metrics
task and future-prediction metrics for stronger claims
tool / memory / belief as future task areas, not hand-built core schemas
```

Avoid:

```text
handcrafted ontology as the project center
fixed semantic database as the source of truth
broad schemas that no tokenizer, model, or evaluator consumes
large architectural expansions before evaluation pressure exists
```

## Run Tests

Local development installs PyTorch through the project optional dependency:

```sh
./scripts/setup_local.sh
uv run python -m unittest
```

On RunPod, use an official PyTorch template and keep its system PyTorch/CUDA
stack. Do not run `uv sync`, because it may install a PyTorch wheel that does
not match the host NVIDIA driver. The tested template is `runpod-torch-v220`
(`Runpod Pytorch 2.2.0`, CUDA 12.1).

```sh
./scripts/setup_runpod.sh
python -m unittest
```

When RunPod also needs torchvision, install the wheel that matches the template's
system PyTorch/CUDA stack:

```sh
./scripts/setup_runpod.sh
./scripts/setup_runpod_vision.sh
python -m unittest
```

## Current Training Entrypoints

The current prototype supports four single-task training paths:

```text
text LM
  text corpus -> token sequence -> causal text model

image classification
  image -> class label

image-text choice
  image + candidate texts -> selected candidate

image-text answer
  image + prompt -> answer tokens
```

The image-text choice and image-text answer paths are different output forms,
not a temporary/permanent hierarchy. Choice is useful for matching, retrieval,
and multiple-choice tasks. Answer is useful for token-generating image/text
tasks.

The shared multimodal model is a shell with task-specific routes and heads over
one shared Transformer core. A task may leave some routes unused; unused
tokenizer, text embedding, image input, token output, choice scoring, or
classification components do not make that task secondary.

Text language modeling can train a tokenizer by default, but the preferred
workflow is to train a text tokenizer once and reuse it across text-consuming
tasks:

```sh
uv run python -m intrep.train_text_tokenizer \
  --corpus-path data/external/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --tokenizer-vocab-size 1024
```

```sh
uv run python -m intrep.train_language_model \
  --corpus-path data/external/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --metrics-path runs/text.json \
  --checkpoint-path runs/text.pt
```

FineWeb-Edu can be sampled into a local text corpus before training. This
command requires the Hugging Face `datasets` package in the active environment:

```sh
python -m intrep.prepare_fineweb_edu_text \
  --output-path data/external/fineweb_edu_sample.txt \
  --max-bytes 1000000
```

Image classification uses image patch embeddings, the shared Transformer core,
and a classification head:

```sh
uv run python -m intrep.cifar10_image_corpus \
  --batch-path data/cifar-10-batches-py/data_batch_1 \
  --output-path runs/cifar10-train.jsonl \
  --image-output-dir runs/cifar10-train-images

uv run python -m intrep.train_image_classification \
  --train-path runs/cifar10-train.jsonl \
  --metrics-path runs/cifar10.json \
  --checkpoint-path runs/cifar10.pt
```

The same training command can read torchvision-style ImageFolder datasets:

```sh
uv run python -m intrep.train_image_classification \
  --train-image-folder data/images/train \
  --eval-image-folder data/images/eval \
  --image-size 224 224 \
  --metrics-path runs/image-folder.json \
  --checkpoint-path runs/image-folder.pt
```

Image-text choice trains a shared multimodal model to score candidate text
answers:

```sh
uv run python -m intrep.train_image_text_choice \
  --train-path runs/fashion-choice-train.jsonl \
  --eval-path runs/fashion-choice-eval.jsonl \
  --tokenizer-path runs/text-tokenizer.json \
  --prompt "What is this item?" \
  --metrics-path runs/fashion-choice.json \
  --checkpoint-path runs/fashion-choice.pt
```

Image-text answer trains the token output path from image plus prompt to answer
tokens:

```sh
uv run python -m intrep.train_image_text_answer \
  --train-path runs/fashion-answer-train.jsonl \
  --tokenizer-path runs/text-tokenizer.json \
  --metrics-path runs/fashion-answer.json \
  --checkpoint-path runs/fashion-answer.pt
```

Shared multimodal training commands also accept `--init-checkpoint-path` for
compatible shared multimodal checkpoints. Text-consuming commands accept
`--tokenizer-path` to reuse a fixed tokenizer; if both a checkpoint and a
tokenizer path are provided, the explicit tokenizer path is used. Checkpoint
initialization loads compatible model weights independent of the source task
name.

Image classification uses the same shared multimodal model shell, with the
image route, shared Transformer core, and classification head active.
Its checkpoint restores the model state for that training run. Reusing a
checkpoint for another run is a compatibility question about the components
that the next task needs, not about which task produced the checkpoint.

IDX datasets such as MNIST and Fashion-MNIST, and CIFAR-10 python batches, can
produce these image JSONL forms:

```text
image-classification JSONL
image-text-choice JSONL
image-text-answer JSONL
```
