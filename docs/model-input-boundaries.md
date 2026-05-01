# Model Input Boundaries

This document defines the current input-boundary vocabulary for model training.
It is not a complete ontology of every possible data type. Its purpose is to
keep the project from mixing raw data, token IDs, embedding vectors, Transformer
hidden states, and task outputs into one unclear schema.

The project should not force every modality into one raw data format. A
modality is a kind of input or output form, such as text, image, video, audio,
discrete action, tool result, or label text. Raw examples should stay close to
the dataset or interaction they come from.

Commonization should happen at the input embedding sequence consumed by the
Transformer core.

```text
raw examples
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
  -> hidden states
  -> task-specific output layer and objective
```

## Raw Examples

Raw examples are task-level data items before tokenization or embedding.

Examples:

```text
language modeling:
  text

image choice:
  image_path
  choices
  answer_index

image/text choice:
  image_path
  prompt_text when needed
  choices
  answer_index

video or audio tasks:
  media path or reference
  task-specific text, labels, or feedback when needed
```

The raw example shape should be chosen for the task and source data. Do not add
generic cross-task fields just to make unrelated tasks look uniform. Add
structure only when a tokenizer, input layer, training objective, or evaluator
actually uses it.

## Input Layers

Input layers are the modality-specific path from raw data to model input
vectors.

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

The current image choice path uses this concrete form:

```text
image-choice JSONL
  -> local PGM image
  -> ImagePatchInputLayer
  -> candidate text tokenizer / embedding layer
  -> input embedding sequence
  -> SharedTransformerCore
  -> candidate score
```

Other modalities can use their own input layers. A full text, image, audio, or
video encoder may include both input layers and a Transformer or CNN body. This
document names the boundary explicitly because `encoder` is often used at
multiple levels of granularity.

## Input Embedding Sequence

The cross-modal common model input is the input embedding sequence: a continuous
vector sequence with shape `[batch, sequence, hidden]`.

This is the layer where text token embeddings, image patch embeddings, and
other modality-specific embeddings can meet. Token IDs are not the common
cross-modal representation; they must first become embedding vectors before
entering the Transformer core.

## Hidden States

The shared Transformer core consumes input embedding sequences and produces
hidden states.

```text
input embedding sequence
  -> shared Transformer core
  -> hidden states
```

Hidden states are also representations, but they are not the same boundary as
input embeddings. The distinction matters:

```text
input embeddings:
  vectors before the Transformer core

hidden states:
  contextual vectors after the Transformer core
```

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

## Output Layers and Objectives

Output layers and objectives may remain task-specific.

```text
hidden states
  -> classification head / text decoder / candidate scoring / value head
  -> loss or evaluation
```

Selection classification can be implemented with a classification head, a text
decoder, candidate scoring, or another objective. The important boundary is that
each input modality reaches the shared core through an appropriate input layer,
not through a forced raw-data schema.
