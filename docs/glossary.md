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

### Problem Package

A problem package groups code that is tightly tied to one problem-oriented model
surface and one input/target/output shape, such as model input construction,
output heads, losses, metrics, and evaluation.

Use `task` informally when convenient, but do not use it as a package boundary.
Use narrower terms such as `Problem`, `Training Example`, `Sample`,
`Objective`, or `Loss` when that distinction matters.

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

### Evidence

Evidence is stored information that can support later target construction,
evaluation, or interpretation, but is not itself a problem-specific target.

Use this term when the same stored signal can be interpreted in multiple ways by
different problems or objectives.

Examples:

- engine score lines for a shogi position
- MCTS visit counts for a position
- a teacher's ranked moves
- repeated observations of the same position or transition

Evidence can later be turned into a target by a problem-specific rule. For
example, the same shogi engine scores may become a policy distribution, a value
target, a ranking target, or simply evaluation metadata.

Do not use `Evidence` when a narrower term is clearer. For actual world
interaction, use `Experience`. For the optimized training signal, use `Target`.

### Dictionary

A Dictionary is an exact key-to-value mapping.

Use this term when the intended operation is direct lookup:

`key -> value`

Examples:

- position SFEN -> cached best move
- document id and span -> stored label
- content hash -> cached result

A Dictionary is narrower than an Index. It implies exact keys and a relatively
direct returned value.

### Index

An Index is a lookup or search structure used to find stored entries or
references.

An Index may behave like a Dictionary when the lookup is exact, but it can also
support multiple matches, filtering, ranking, or approximate retrieval.

Examples:

- exact position index keyed by SFEN
- inverted text index
- vector similarity index
- content-hash index

Use `Index` when the important concept is finding or referring to stored
information, not merely owning source records. Do not use `Index` as a synonym
for every durable store.

### Shogi Position Index

A Shogi Position Index is a shogi-specific Index keyed by `position_sfen`.

It is for looking up known entries for a position. It is not an Experience
Store: game records and transitions describe what happened, while the position
index describes what is known for a position key.

The first implementation is exact lookup. Approximate lookup for similar
positions is a separate design question.

### Engine Analysis

Engine Analysis is analysis produced by an engine for a position or state.

In the current shogi code, `ShogiEngineAnalysis` is intentionally narrow: it
stores shogi-engine analysis for a shogi position. It is not a generic
annotation framework and not a training target.

USI info emitted during the recorded action decision belongs to the shogi game
transition record, not to `ShogiEngineAnalysis`.

Engine Analysis may later be used as Evidence by a problem-specific target
construction rule.

### MCTS Leaf Parallelization

MCTS Leaf Parallelization is a Monte Carlo Tree Search parallelization pattern
where one search tree selects multiple pending leaf nodes before neural
evaluation and backup.

It is different from running multiple games at the same time. It is also
different from the neural evaluation batch size:

- parallel games: multiple games or root positions are active at once
- MCTS leaf parallelization: one MCTS tree has multiple in-flight leaf
  selections
- evaluation batch size: the maximum number of selected leaf positions sent to
  the model in one forward pass

Without an in-flight marker such as virtual loss or virtual visits, multiple
parallel selections can choose the same promising path or even the same leaf,
because the search tree statistics have not yet been updated by backup.

The current shogi self-play implementation batches across active games. It does
not implement MCTS leaf parallelization within one tree.

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

### Experience Store

An Experience Store is durable source storage for generated or collected
experience.

It is not a PyTorch Dataset. Training should use explicit Data Selection or a
fixed Training Data Bundle derived from the store.

Experience Store records should remain close to the source experience and
should not be reshaped around one objective, run, or model.

Experience Store is a concept, not proof that a shared implementation exists.
The current implementation is shogi-local. Whether Experience Store storage
should remain world/source-specific or become a shared abstraction is an open
design question.

### Online Experience Replay

Online Experience Replay is a training-time reinforcement-learning method where
new experience is added during learning and older experience is sampled again
for model updates.

This term implies a dynamic learner-facing component. It is the right concept
when training samples repeatedly from changing experience, often through a
Replay Buffer.

### Replay Buffer

A Replay Buffer is the dynamic storage / sampling component used for Online
Experience Replay.

It usually supports appending new experience, sampling training batches, and
some policy for capacity, recency, priority, or replacement.

Do not use this term for a static file or for one-time Training Data Bundle
construction.

### Offline Experience Reuse

Offline Experience Reuse means using previously collected experience records to
build a fixed training or evaluation set before training starts.

This is closer to ordinary data selection than to a Replay Buffer. It may use
self-play records, teacher records, run outputs, or an Experience Store as
sources, but the result is a fixed Training Data Bundle or PyTorch Dataset input.

The training pipeline is not special: once fixed, it is learned from like an
ordinary dataset. The distinction is that the selected source records are
experience records, meaning they come from interaction with a world and may
carry actions, rewards, outcomes, actor identity, or trajectory context.

### Training Data Bundle

A Training Data Bundle is a materialized, fixed collection of training and
evaluation inputs derived from source records.

It is not source data and not a PyTorch `Dataset`. It may contain selected
records, split files, target-source metadata, data selection metadata, and a
manifest. The bundle is the artifact a training run can consume without relying
on run-local generated files as the source of truth.

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

A sample is one runtime item consumed by training or evaluation. In PyTorch
code, it is usually the item returned by `Dataset.__getitem__` and batched by a
`DataLoader`.

A sample may be built from a training example, but it may also contain tensors,
masks, encoded forms, targets, weights, or metadata needed by the training or
evaluation loop.

### Sample Schema

A Sample Schema defines the fields and meanings of a sample. Samples with the
same schema can usually share collation, model routing, loss, and metrics.

It may include model inputs, targets, masks, weights, and metadata. It is
broader than a model input schema.

The schema is semantic, not only structural. Two samples with the same tensor
shapes may still have different schemas if their fields mean different things.

A dataset may contain one Sample Schema or multiple Sample Schemas. Mixed-schema
datasets are allowed conceptually, but they require explicit routing,
collation, loss, and metric handling.

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

### Target Construction

Target Construction is the problem-specific rule that turns selected source
records, evidence, stored targets, or feedback into Targets.

It is separate from Data Selection. Data Selection decides what material is
included. Target Construction decides how included material becomes the value
that a model output is compared with.

Examples:

- selected move -> one-hot policy target
- engine MultiPV scores -> policy distribution target
- game winner -> return target
- engine score -> value target

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

### Training Configuration

A Training Configuration defines how a training run uses data, model parts,
objectives, losses, optimization, and train/freeze policy.

It may specify which model parts are trained, frozen, adapted, or replaced by
precomputed representations. It is not a Dataset, Sample Schema, Model, or run
output.

### Loss

A loss is the numeric quantity optimized for an objective during training.
It is one way to make an objective trainable.

Examples:

- cross entropy
- mean squared error
- binary cross entropy
