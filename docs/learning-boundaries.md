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

## Data Selection and Runs

Data Selection states which source records and target material are included for
a declared use.

Experience Stores are source storage for generated or collected experience.
They are not PyTorch Datasets and should not be the direct learning boundary.
Training should use explicit Data Selection or a fixed Training Data Bundle
built from a declared Data Selection.

PyTorch `Dataset` objects should stay thinner than Data Selection. They adapt
an already-selected training or evaluation set into indexed samples; they should
not be the source of truth for target generation, split policy, or learning
intent.

A run produces artifacts such as raw logs, example caches, metrics, or
checkpoints. Runs can provide material for future Data Selection, but a run
artifact is not itself the selection boundary.

Training should be explainable by Data Selection, training-example meaning, and
a training configuration, not only by a convenient artifact path.

## Model Inputs

Training and inference should not treat raw source records as direct model
inputs. Problem-specific code adapts selected records or training examples into
model-ready inputs.

## Outputs and Objectives

Model outputs are interpreted by objectives and evaluators. Different
objectives can be built from similar source records.

Reinforcement learning is different because the target is not just a fixed
label from the record. It optimizes behavior under feedback. That difference
belongs in the objective and execution loop, not in a universal raw schema.

## Shogi RL Artifact Boundary

For shogi-specific source-side versus policy/value-problem ownership, see
[Shogi Learning Boundaries](shogi/learning-boundaries.md).

The current shogi RL loop uses a CLI/subprocess and artifact boundary between
`intelligence-representation` and `shogi-arena-agent`.

`intelligence-representation` owns the learning loop: checkpoint selection,
raw game-record ingestion, replay or fixed training-data construction, model
updates, metrics, and checkpoint promotion.

`shogi-arena-agent` owns shogi game generation runtime: player construction,
USI engine processes, runtime move selection, search settings, game execution,
and raw game-record JSONL output.

Shogi player-vs-player match entrypoints belong in `shogi-arena-agent`.
`intelligence-representation` should pass checkpoints and read game-record
artifacts, not mirror match runner CLIs.
RunPod wrappers for player-vs-player matches follow the same rule.

The artifact contract is:

- checkpoint files flow from `intelligence-representation` to
  `shogi-arena-agent`
- shogi game-record JSONL flows from `shogi-arena-agent` back to
  `intelligence-representation`
- generated records carry actor metadata such as checkpoint identity, move
  selector, and search settings for later explanation and selection
- evaluation metrics and game records belong to the side that runs the
  evaluation

Do not make `intelligence-representation` import `shogi-arena-agent` internals
only to run self-play. Keep the subprocess boundary until measured overhead or
schema coordination makes a smaller shared library boundary clearly better.

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
