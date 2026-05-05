# Glossary

This document records working terms for data and model boundaries. It is not a
complete ontology. Keep terms narrow enough to guide code and package
responsibilities.

## Boundary Chain

```text
source record
  -> training example
  -> model input

object or world state in a model input
  -> form
  -> encoding
  -> input embedding
  -> hidden state
  -> output head
  -> model output

model output + target
  -> objective
  -> loss
```

## World

A world is a structured setting with state, entities, relations, changes,
constraints, and consequences. Some worlds are interactable through actions;
others are observed or replayed through source records.

Examples:

- grid simulation
- shogi game environment
- browser or tool environment
- physical or sensor environment
- replayed trajectory or game record

A world-like interface exposes observations or transitions from a source record
without necessarily supporting free interaction or branching.

In agentic settings, an agent is not only an observer. It has a viewpoint,
action interface, history, constraints, and identity, and its state or outputs
can become part of the world.

## Source Record

A source record is a stored source item before it is cut or transformed for a
specific objective.

Examples:

- shogi game record
- grid episode record
- image file and label row
- text document
- browser interaction log

A source record can produce one or more training examples.

## Experience

Experience is a source record produced by interaction with a world. It is
usually ordered over time and may contain observations, actions, feedback,
rewards, or consequences.

Examples:

- grid episode trajectory
- browser interaction log
- tool-use trace
- self-play game record

Do not force static datasets to be experience. An image file and label row can
be a source record without being experience.

## Training Example

A training example is a unit made from a source record for a specific
objective. The source record may be static data or experience.

Examples:

- position and selected move
- position and winner target
- grid state, action, and next grid state
- image and class label
- text window and next token targets

## Model Input

A model input is the input-side value passed to a model for training,
evaluation, or inference.

Model input is a role, not a representation type. It may contain forms,
encodings, tensors, masks, candidate sets, or already-built input embeddings.

## Form

A form is an external shape used to store, exchange, display, or prepare an
object or world state. It does not have to be human-readable.

Examples:

- shogi: SFEN, KIF, USI moves, rendered board images, game records
- grid worlds: ASCII grids, tensors, transitions, rendered images
- images: PNG, PGM, pixels, patches, labels, captions
- text: raw text, bytes, rendered layout

## Encoding

An encoding is a form converted into a discrete or numeric structure for a
specific processor, model, or objective. It is model-facing, but it is not yet a
learned vector representation.

Examples:

- text token IDs
- board token IDs
- move feature IDs or candidate feature matrices
- normalized image tensors
- grid tensors, masks, and offsets

Cached token IDs or tensors are still encodings. They may have a file format on
disk, but their role in the pipeline is encoded model input preparation.

## Input Embedding

An input embedding is a learned continuous vector sequence produced from an
encoding before it enters the model core.

Examples:

- token embeddings
- square, piece, side-to-move, or move embeddings
- image patch embeddings
- action embeddings

Input embeddings are the entry point to learned representation. They are
distinct from token IDs, feature IDs, or raw numeric tensors.

The word `embedding` by itself is broader in common machine-learning usage and
can also describe sentence, image, or retrieval vectors. This glossary avoids
using bare `embedding` as a project boundary term. Prefer concrete names such
as `input embedding`, `token embedding`, `patch embedding`, or `move embedding`.

## Hidden State

A hidden state is a contextual vector produced by a model core from embeddings.
Transformer hidden states are the main current example.

Hidden states are learned representations, but this glossary keeps the term
`hidden state` for the concrete model boundary and reserves `representation` for
broader discussion.

## Output Head

An output head is an output-side module that reads hidden states and produces a
model output.

Examples:

- classification head
- language-modeling head
- candidate scoring head
- policy head
- value head

This glossary uses `output head` rather than bare `head` to avoid confusion with
attention heads inside Transformer layers or read/write heads in memory models.

## Model Output

A model output is the direct value returned by an output head or decoder.

Examples:

- class logits
- next-token logits
- candidate scores
- value estimates
- reward estimates
- predicted tensors

Model output is a role, not a representation type. It may describe a form, an
encoding, a scalar, a tensor, or a distribution over choices.

## Target

A target is the expected value or teacher signal used to evaluate a model
output during training or evaluation.

Examples:

- class label
- next token ID
- selected move
- next grid state
- winner or value target

## Objective

An objective defines what should count as a good prediction. It says how model
outputs and targets should be interpreted for training or evaluation.

Examples:

- next-token prediction
- candidate move selection
- image classification
- next-state prediction
- value prediction

## Loss

A loss is the numeric quantity optimized for an objective during training.

Examples:

- cross entropy
- mean squared error
- binary cross entropy

## Representation

Representation is the broadest term. It can refer to any information-carrying
shape used by the project, including forms, encodings, input embeddings, hidden
states, labels, and learned features.

When discussing code boundaries, prefer the narrower terms above. Use
`representation` when the broader idea is intentional.
