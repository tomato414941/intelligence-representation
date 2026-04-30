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

docs/legacy/
  Historical notes and retired experiments
```

## Canonical Docs

Read these first:

- [Concept](docs/concept.md)
- [Predictive Representation System](docs/predictive-representation-system.md)
- [Model Input Boundaries](docs/model-input-boundaries.md)
- [Learning and Execution](docs/learning-and-execution.md)
- [World Model Centering](docs/world-model.md)
- [Bitter Lesson Correction](docs/bitter-lesson.md)
- [Evaluation](docs/evaluation.md)
- [Current Results](docs/current-results.md)

Historical notes:

- [Legacy Docs](docs/legacy/)

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

```sh
uv sync
uv run python -m unittest
```

## Current Training Entrypoints

Text language modeling uses the main byte-pair tokenizer by default:

```sh
uv run python -m intrep.train_language_model \
  --corpus-path data/external/tiny-shakespeare.txt \
  --metrics-path runs/text.json \
  --checkpoint-path runs/text.pt
```

Fashion-MNIST-style image classification uses image patch embeddings, the
shared Transformer core, and a classification head:

```sh
uv run python -m intrep.evaluate_fashion_mnist \
  --train-path runs/fashion-train.jsonl \
  --eval-path runs/fashion-eval.jsonl \
  --metrics-path runs/fashion.json
```
