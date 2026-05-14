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
  -> problem-specific output layer
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

legacy/
  Historical archive outside the active implementation path
```

## Canonical Docs

Read these first:

- [Concept](docs/concept.md)
- [Predictive Representation System](docs/predictive-representation-system.md)
- [Model Boundaries](docs/model-boundaries.md)
- [Learning Boundaries](docs/learning-boundaries.md)
- [Worlds and Experience](docs/worlds-and-experience.md)
- [Datasets](docs/datasets.md)
- [RunPod](docs/runpod.md)
- [Compute Costs](docs/compute-costs.md)
- [World Model Centering](docs/world-model.md)
- [Bitter Lesson Correction](docs/bitter-lesson.md)
- [Evaluation](docs/evaluation.md)
- [Evidence](docs/evidence.md)

## Design Constraints

Prefer:

```text
raw examples before premature schemas
problem-specific input layers
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

For RunPod setup, image, CUDA/PyTorch, torchvision, region, and fallback notes,
see [RunPod](docs/runpod.md).

## Current Training Entrypoints

The current prototype includes these training entrypoints:

```text
intrep.train_text_tokenizer
intrep.train_language_model
intrep.train_image_classification
intrep.train_image_text_choice
intrep.train_image_text_answer
intrep.train_grid_step_prediction
intrep.train_shogi_policy_value
```

Problem models compose task-specific input layers, the shared Transformer core
where useful, and task-specific output heads. See [Model Boundaries](docs/model-boundaries.md)
and [Learning Boundaries](docs/learning-boundaries.md) for the current design.

Text language modeling can train a tokenizer by default, but the preferred
workflow is to train a text tokenizer once and reuse it across text-consuming
tasks:

```sh
uv run python -m intrep.train_text_tokenizer \
  --corpus-path data/tiny-shakespeare/raw/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --tokenizer-vocab-size 1024
```

```sh
uv run python -m intrep.train_language_model \
  --corpus-path data/tiny-shakespeare/raw/tiny-shakespeare.txt \
  --tokenizer-path runs/text-tokenizer.json \
  --metrics-path runs/text.json \
  --checkpoint-path runs/text.pt
```

FineWeb-Edu can be sampled into a local text corpus before training. This
command requires the Hugging Face `datasets` package in the active environment:

```sh
python -m intrep.text.prepare_hf_text_slice \
  --output-path data/external/fineweb_edu_sample.txt \
  --max-bytes 1000000
```

Image classification uses image patch embeddings, the shared Transformer core,
and a classification head:

```sh
uv run python -m intrep.vision.cifar10_corpus \
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
