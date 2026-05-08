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
  -> input embedding module
  -> input embedding
  -> hidden state
  -> output head
  -> model output

model output + target
  -> objective
  -> loss
```

## Package-Level Terms

### Source-Side Package

A source-side package holds source records, forms, IO, encodings, conversions,
input preparation utilities, or world-like utilities for a source family or
representation family.

Current source-side packages remain intentionally narrow:

- `vision/` and `text/` are form/input-oriented.
- `worlds/shogi/` and `worlds/grid/` are world-oriented.

Do not use `domain` as the umbrella term for these packages. They are not all
the same kind of category.

### Task Package

A task package is an objective-bound model surface. It can bind model input
construction, output heads, targets, losses, metrics, training, checkpointing,
and evaluation for a task family.

Examples:

- image classification
- language modeling
- grid step prediction
- shogi move choice

## Interaction Terms

### World

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

### Observation

An observation is a view, signal, or recorded appearance made available from a
world to an actor, model, or recorder.

Observation is a role, not a representation type. An observation can be carried
by many forms or encodings, and it may expose only part of the world.

Examples:

- visible grid from a grid world
- current board position in a shogi game
- browser page view, screenshot, DOM, or accessibility tree
- sensor reading from a physical environment

### Action

An action is an intervention by an actor, model, agent, or policy that can
affect a world, world-like interface, or the actor's own state.

Action is a role, not a representation type. It can be carried by forms or
encodings, and it can appear as model input, target, or model output depending
on the objective.

Examples:

- shogi move
- grid movement command
- browser click or typed text
- tool call
- generated response that changes later context
- memory, context, planning, or self-evaluation update

### Feedback

Feedback is information returned from a world, evaluator, user, or process that
can guide learning, evaluation, correction, or future behavior.

Feedback is a role, not a representation type. It can be carried by forms or
encodings, and it can appear in experience, targets, metrics, or future context.

Examples:

- reward
- win/loss
- correction
- label
- user rating
- error message
- tool result status

## Data Pipeline Terms

### Source Record

A source record is a stored source item before it is cut or transformed for a
specific objective.

It preserves source-side meaning. It should not be reshaped only to fit one
objective, model, or run.

Examples:

- shogi game record
- grid episode record
- image file and label row
- text document
- browser interaction log

A source record can produce one or more training examples.

### Experience

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

### Data Selection

Data Selection decides which source records, training examples, or stored
targets are included for a declared use. It is about inclusion, not example
construction, target generation, sampling, or optimization.

### Training Example

A training example is an objective-specific unit that defines what is used as
model input and what is used as target or feedback.

It may be made from a source record, experience, stored target, or a combination
of them.

`Training Example` is the project term for this meaning-level unit. `Example`
is acceptable as a short code name when the context is already clear. A
training example may be stored, but storage does not make it the source record.

Examples:

- position and selected move
- position and winner target
- grid state, action, and next grid state
- image and class label
- text window and next token targets

### Sample

A sample is the runtime item returned by `Dataset.__getitem__`. It is the
PyTorch-side item that can be batched by a `DataLoader`.

A sample may be built from a training example, but it may already contain
tensors, masks, encoded forms, or metadata needed by the training loop.

### PyTorch Dataset

A PyTorch `Dataset` is an adapter that returns indexed samples for training or
evaluation. It is not the source of truth for raw data, target generation,
split policy, or learning intent.

### Model Input

A model input is the input-side value passed to a model for training,
evaluation, or inference.

Model input is a role, not a representation type. It may contain forms,
encodings, tensors, masks, candidate sets, or already-built input embeddings.

## Representation Boundary Terms

### Form

A form is an external shape used to store, exchange, display, or prepare an
object or world state. It does not have to be human-readable.

Examples:

- shogi: SFEN, KIF, USI moves, rendered board images, game records
- grid worlds: ASCII grids, tensors, transitions, rendered images
- images: PNG, PGM, pixels, patches, labels, captions
- text: raw text, bytes, rendered layout

### Encoding

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

### Input Interface

An input interface is the broader route by which source-side or world-side
values become model inputs. It may include IO, decoding, tokenization,
encoding, or input embedding modules.

Use narrower terms such as `encoding` or `input embedding module` for code
boundaries when possible.

### Input Embedding Module

An input embedding module is a model-side module that converts model inputs or
encodings into input embedding sequences for the shared core.

Use this term when precision matters. Use `input interface` for the broader
route from source-side values toward model inputs. `Input layer` is acceptable
in informal discussion, but it is too broad as a project boundary term.

### Input Embedding

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

### Hidden State

A hidden state is a contextual vector produced by a model core from input
embeddings. Transformer hidden states are the main current example.

Hidden states are learned representations, but this glossary keeps the term
`hidden state` for the concrete model boundary and reserves `representation` for
broader discussion.

### Representation

Representation is the broadest term. It can refer to any information-carrying
shape used by the project, including forms, encodings, input embeddings, hidden
states, labels, and learned features.

When discussing code boundaries, prefer the narrower terms above. Use
`representation` when the broader idea is intentional.

## Output And Learning Terms

### Output Head

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

### Model Output

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

### Target

A target is the value a model output is compared with or interpreted against
for training or evaluation. A target is a role: it may be a source record, part
of a source record, a derived value, or an externally produced teacher signal.

Examples:

- class label
- next token ID
- selected move
- next grid state
- winner or value target

### Objective

An objective is the goal to optimize or evaluate. It defines what behavior
counts as good and how model outputs, targets, or feedback are interpreted for
training or evaluation.

Examples:

- next-token prediction
- candidate move selection
- image classification
- next-state prediction
- value prediction

### Loss

A loss is the numeric quantity optimized for an objective during training.
It is one way to make an objective trainable.

Examples:

- cross entropy
- mean squared error
- binary cross entropy
