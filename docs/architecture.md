# Architecture

This document records the current package responsibility boundaries. It is a
short orientation document, not a complete design spec.

## Project Direction

The project is not trying to force every source into one common raw data
schema. It should support many ways to represent the same object or world.

Examples:

- shogi as board tokens, move candidates, game records, text, or rendered images
- grid worlds as tensors, transitions, text, or rendered images
- images as pixels, patches, labels, captions, or text answers
- text as raw text, bytes, token IDs, or rendered layout when needed

Commonization should happen where it helps learning and comparison, especially
around input embedding sequences, shared cores, task models, and transfer. Raw
records should stay close to their source and task.

## Terms

Use [Glossary](glossary.md) for the current boundary terms.

## Package Responsibilities

- `core/`: domain-agnostic representation computation and shared utilities.
- `vision/`, `text/`: source-side packages for form/input-oriented external
  forms, encodings, IO, and conversions.
- `worlds/`: source-side packages for world-oriented records, replay,
  observations, actions, transitions, encodings, and world-like utilities.
  Current packages include `worlds/shogi/` and `worlds/grid/`.
- `tasks/`: task packages for objective-bound model surfaces that bind model
  input construction, shared cores, output heads, targets, losses, metrics,
  checkpoints, training, or evaluation.
- `transfer/`: reuse of learned state across task models.

Do not introduce `domains/` as an umbrella package. The source-side packages
are not all the same kind of category. `forms/` remains deferred until a
concrete form/input-oriented boundary problem needs it.

## Task Layer

`tasks/` is not for dataset instances such as MNIST, CIFAR-10, or one shogi
corpus run. It is for task families such as image classification, image-text
choice, language modeling, retrieval, grid step prediction, and shogi move
choice.

Dataset-specific configuration, run settings, and generated artifacts should
stay outside the package code unless they become reusable source definitions.

## Multi-Task Learning

The structure should not prevent future multi-task learning. Avoid splitting a
task into many tiny files just to mirror an abstract ontology. A task package
may keep model, loss, metrics, checkpoint, and training code together until a
real repeated pattern needs extraction.

If multi-task learning becomes active work, prefer small shared boundaries such
as per-task batch-loss/evaluation entry points over a large common schema for
all examples.
