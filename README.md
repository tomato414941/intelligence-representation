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
```

## Canonical Docs

Read these first:

- [Concept](docs/concept.md)
- [Predictive Representation System](docs/predictive-representation-system.md)
- [Model Boundaries](docs/model-boundaries.md)
- [Learning Boundaries](docs/learning-boundaries.md)
- [Worlds and Experience](docs/worlds-and-experience.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
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

The current prototype includes text, image, image/text, grid, and shogi
training entrypoints. See [Training](docs/training.md) for command examples,
tokenizer reuse, and checkpoint initialization notes.

```text
intrep.train_text_tokenizer
intrep.train_language_model
intrep.train_image_classification
intrep.train_image_text_choice
intrep.train_image_text_answer
intrep.train_grid_step_prediction
intrep.train_shogi_policy_value
```

Dataset preparation notes live in [Datasets](docs/datasets.md).
