# Representation, Signal, and Symbol

## Position

This document defines the conceptual boundary between representation, signal,
symbol, token, modality, and format in this project.

It is not a data format specification. In particular, "signal" here means a
conceptual appearance that can be received, generated, predicted, or acted on.
It does not mean the retired `Signal` class.

The active engineering direction is:

```text
raw examples or interaction records
  -> modality-specific input layers
  -> input embedding sequence
  -> shared Transformer core
  -> task-specific output layer and objective
```

The project should not make raw data look uniform by wrapping everything in a
generic envelope. Commonization should happen at the model-facing
representation boundary, not at the raw-data boundary.

## Core Terms

Representation is the broadest term in this document.

```text
representation:
  something that carries information for a model, system, or evaluator
  may be raw, discrete, continuous, symbolic, or latent
  includes tokens, embeddings, hidden states, labels, text, and learned features
```

A signal is an appearance that a system can receive, generate, predict,
remember, or act upon.

```text
signal:
  input, output, observation, feedback, or generated appearance
  may be continuous or discrete
  does not imply a shared raw schema
```

A symbol is a relatively discrete or interpretable unit that can stand for
something.

```text
symbol:
  can refer, describe, command, classify, or explain
  is often text-like or label-like
  may be learned rather than manually assigned
```

A token is a discrete model-facing unit produced by a tokenizer.

```text
token:
  discrete pre-embedding unit
  common for text and other intentionally discrete inputs
  not the universal cross-modal representation
```

The relation is:

```text
representation:
  broad model/system-facing information carrier

signal:
  representation as an incoming, outgoing, or feedback appearance

symbol:
  representation with relatively discrete, interpretable, referential use

token:
  discrete unit before embedding, usually produced by a tokenizer
```

Natural language is both signal and symbol. As model input, it can become
tokens and then token embeddings. Images and audio are usually signals at the
raw level, but not symbols until a model, label, caption, or learned structure
makes them symbol-like.

## Representation Principle

The project should keep raw examples close to their source task.

```text
text:
  raw text

image choice:
  image_path
  choices
  answer_index

audio transcription:
  audio_path
  transcript or answer text

interaction:
  observation
  action or model output
  feedback, reward, or tool result
```

These examples should become comparable only after modality-specific input
layers have done the necessary work.

```text
text example
  -> tokenizer
  -> token ids
  -> token embeddings
  -> input embedding sequence

image example
  -> image loader
  -> patch embedding layer
  -> input embedding sequence

image-to-text example
  -> image input layer + text tokenizer / embedding layer
  -> input embedding sequence
```

This keeps the system simple. Do not add generic cross-task fields just to make
unrelated tasks appear uniform.

## Boundaries Are Earned

A boundary name is not an ontology. It is a practical route for a class of data
when that route helps implementation, learning, or evaluation.

Useful boundaries can include:

```text
modality-specific input layer
tokenizer selection
embedding layer
decoder or output head
sequence window construction
objective target selection
evaluator selection
```

Such a boundary may reflect modality:

```text
text
image
audio
video
```

It may also reflect an interaction role when an experiment actually needs it:

```text
observation
action
feedback
answer
tool result
```

But the boundary must earn its place. A name should not become a field or class
unless an input layer, dataset, evaluator, objective, or execution loop uses it.

## Format Is Not Modality

Concrete encodings and file formats are loading concerns.

```text
format:
  utf-8
  markdown
  png
  jpeg
  wav
  mp4
```

They should not automatically become conceptual categories. A PNG and a JPEG
may both be image inputs. A Markdown file and a JSON file may both be text
inputs if the input layer treats them as text.

## Text As Reference Case

Text is the reference success case because it turns naturally into long,
learnable token sequences with abundant data.

```text
structured text
  -> tokenizer
  -> token ids
  -> token embeddings
  -> next-token or continuation objective
```

Other modalities should be judged by a more general practical question:

```text
Can this raw example or interaction record become a learnable model input
sequence with a clear objective or evaluator?
```

They do not need to resemble natural language externally. Image patches, audio
frames, action records, and feedback can all be valid if they create
model-facing representations that a shared core can learn from.

## Observation

`observation` is risky as a project-wide abstraction because it names a
relation between a system and an incoming signal, not a concrete modality.

```text
input signal ~= observation
```

This differs from concrete signal forms:

```text
text:
  modality / input-layer family

image:
  modality / input-layer family

observation:
  relation between a system and an incoming signal
```

Input is necessary, but an `Observation` class, field, or universal record type
is not automatically necessary. Natural language models receive input text
without usually modeling that input as an explicit `Observation` object.

Use an explicit observation boundary only when an implementation or evaluator
needs to distinguish received data from generated, predicted, remembered, or
action-like data.

## Action

`action` is also not a modality in the same sense as text, image, audio, or
video. It is an intervention-side relation.

```text
observation:
  world -> system

action:
  system -> world
```

Action is harder to replace with a concrete modality. A text sequence can
describe an action, command an action, or become an action when the surrounding
system treats it as executable.

```text
text:
  "move north" as language

action:
  "move north" as command to an environment, tool, body, or controller
```

For this reason, action may deserve an explicit boundary earlier than
observation. Even then, the boundary should be added because an experiment
needs action-conditioned prediction or action generation, not because the
project wants a fixed ontology.

## Implementation Guidance

Prefer:

```text
task-specific raw examples
modality-specific input layers
TokenSequence for text or other discrete pre-embedding inputs
input embedding sequence as the shared model input boundary
hidden states as Transformer outputs
loss masks for separating context from supervised answer tokens
```

Avoid:

```text
generic raw-data envelopes
unused cross-task fields
schema fields that no input layer or evaluator consumes
handcrafted semantic taxonomies
making classification, prediction, memory, and action look identical too early
```

The shared middle of the system should be the model-facing representation:

```text
modality-specific input layer
  -> input embedding sequence
  -> shared Transformer core
  -> task-specific output layer / decoder / objective
```

For text-only language modeling, the input layer may be a tokenizer plus an
embedding table. For image classification, it may be an image patch embedding
layer plus a classification head. For image-to-text or audio-to-text tasks,
input and output layers can differ while the Transformer core remains
shareable.

## Retired Direction

The old `Signal` class and Signal JSONL path are retired from the active
implementation. The class name is reserved only to prevent accidental reuse of
the old abstraction.

That retirement does not mean the word "signal" is forbidden. It means the
project no longer uses a generic `channel` / `payload` envelope as the common
raw data format.

The replacement principle is:

```text
do not commonize raw examples prematurely
commonize the model-facing representation boundary
```
