# Glossary

This document records working terms for data and model boundaries. It is not a
complete ontology. Keep terms narrow enough to guide code and package
responsibilities.

## Boundary Chain

```text
object or world state
  -> form
  -> encoding
  -> input embedding
  -> hidden state
  -> output and objective
```

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

## Representation

Representation is the broadest term. It can refer to any information-carrying
shape used by the project, including forms, encodings, input embeddings, hidden
states, labels, and learned features.

When discussing code boundaries, prefer the narrower terms above. Use
`representation` when the broader idea is intentional.
