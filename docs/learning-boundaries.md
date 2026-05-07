# Learning Boundaries

This document explains how the data and learning terms from
[Glossary](glossary.md) relate to each other. The glossary is the source of
truth for term definitions.

## Source Records

Source records should stay close to their source. They can produce many
training examples for different objectives.

They should preserve source-side meaning instead of being reshaped around one
objective, model, or run.

Do not force static dataset records and interaction records into one generic
schema. They are related because both can produce model inputs, not because they
need the same raw fields.

## Dataset Definitions and Runs

A dataset definition states which source records or training examples are used
for a learning purpose.

PyTorch `Dataset` objects should stay thinner than dataset definitions. They
adapt an already-defined training or evaluation set into indexed samples; they
should not be the source of truth for target generation, split policy, or
learning intent.

A run produces artifacts such as raw logs, example caches, metrics, or
checkpoints. Runs can provide material for a dataset, but a run artifact is not
itself the dataset definition.

Training should be explainable by a dataset definition and a training
configuration, not only by a convenient artifact path.

## Model Inputs

Training and inference construct model inputs from source records or training
examples. The shared model boundary is still the input embedding sequence
described in [Model Boundaries](model-boundaries.md).

## Outputs and Objectives

Model outputs are interpreted by objectives and evaluators. Different
objectives can be built from similar source records.

Reinforcement learning is different because the target is not just a fixed
label from the record. It optimizes behavior under feedback. That difference
belongs in the objective and execution loop, not in a universal raw schema.

## Recursive Execution

Recursive use means model outputs can become part of later inputs.

```text
external input
  -> model input
  -> model output
  -> environment, tool, or user response
  -> next model input
```

This loop is mainly an execution concern. During training, a recorded or
simulated loop can be sliced into windows and objectives.

## Non-Goals

This document does not introduce:

```text
a universal raw-data envelope
generic cross-task fields
an RL runtime
a required agent loop
a claim that every task is sequence learning
```

The project should add those only when experiments require them.
