# Model Boundaries

This document defines the current model-boundary vocabulary for model training.
It is not a complete ontology of every possible data type. Its purpose is to
keep the project from mixing raw data, token IDs, embedding vectors, Transformer
hidden states, and task outputs into one unclear schema.

Use [Glossary](glossary.md) as the source of truth for short boundary-term
definitions. This document explains how those terms appear in model training.

The project should not force every modality into one raw data format. A
modality is a kind of input or output form, such as text, image, video, audio,
discrete action, tool result, or label text. Source records should stay close
to the dataset or interaction they come from.

Commonization should happen at the input embedding sequence consumed by the
Transformer core.

```text
source records
  -> modality-specific input embedding modules
  -> input embedding sequence
  -> shared Transformer core
  -> hidden states
  -> output head
  -> model output
```

## Source Records

Source records are source-level data items before tokenization or embedding.

The source record shape should be chosen for the task and source data. Do not
add generic cross-task fields just to make unrelated tasks look uniform. Add
structure only when a tokenizer, input embedding module, training objective, or
evaluator actually uses it.

## Input Embedding Modules

Input embedding modules are the model-side path from model inputs or encodings
to input embedding sequences.

For text, the path usually has a tokenizer and an embedding layer.

```text
text
  -> tokenizer
  -> token ids
  -> token embedding layer
  -> input embedding sequence
```

For images, the path can use patch embeddings.

```text
image
  -> image loader
  -> patch embedding layer
  -> input embedding sequence
```

Other modalities can use their own input embedding modules. A full text, image,
audio, or video encoder may include both input embedding modules and a
Transformer or CNN body. This document names the boundary explicitly because
`encoder` is often used at multiple levels of granularity.

## Input Embedding Sequence

The cross-modal common model input is the input embedding sequence: a continuous
vector sequence with shape `[batch, sequence, hidden]`.

This is the layer where text token embeddings, image patch embeddings, and
other modality-specific embeddings can meet. Token IDs are not the common
cross-modal representation; they must first become embedding vectors before
entering the Transformer core.

## Hidden States

Hidden states are also representations, but they are not the same boundary as
input embeddings. The distinction matters:

```text
input embeddings:
  vectors before the Transformer core

hidden states:
  contextual vectors after the Transformer core
```

## Shared Transformer Core

The shared Transformer core is the model-side module that consumes input
embedding sequences and produces hidden states.

```text
input embedding sequence
  -> shared Transformer core
  -> hidden states
```

The core does not own tokenization, modality loading, output heads, objectives,
or losses. Text, image, grid, and shogi routes can use different input embedding
modules and output heads while keeping the same core boundary.

## Model Module Composition

A problem model may compose named input embedding modules, a shared core, and
output heads. These are replacement boundaries, but they do not require one
universal model class.

Full checkpoint restore should validate the model identity needed to restore the
whole model safely. Partial reuse should be explicit by module name, not by
accidental state-dict key overlap.

## Token IDs and Loss Masks

Token IDs are discrete pre-embedding units.

```text
token_ids
optional loss_mask
```

They are useful for text and other naturally discrete inputs or objectives.
They are not the universal cross-modal representation.

`loss_mask` marks which token positions contribute to a training loss. For
plain language modeling, every next-token position may be trainable. For a
prompted answer, only the answer tokens may be trainable.

Images, video, and audio should not be turned into fake token IDs just to reuse
the text tokenizer path. If a model intentionally uses a learned discrete
visual or audio tokenizer, that should be treated as a specific modeling choice,
not as the default common layer.

## Output Heads

Output heads read hidden states and produce problem-specific model outputs.

```text
hidden states
  -> classification head / text decoder / candidate scoring / value head
  -> model output
```

Selection classification can be implemented with a classification head, a text
decoder, candidate scoring, or another output head. Objectives, losses, and
evaluation read model outputs outside the output-head boundary.
