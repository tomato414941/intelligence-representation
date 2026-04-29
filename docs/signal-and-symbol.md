# Signal, Symbol, and Tokenization

## Position

This document defines the conceptual boundary between signal, symbol, token,
and channel-like implementation boundaries in this project.

It is not a data format specification. In particular, "signal" here means an
abstract learning object, not the retired `Signal` class.

The active engineering direction is:

```text
raw example
  -> tokenizer / encoder / adapter
  -> token sequence or hidden sequence
  -> shared predictive model
```

The project should not make raw data look uniform by wrapping everything in a
generic envelope. The common layer should appear after tokenization or
encoding, not before it.

## Core Terms

A signal is a bounded structured appearance that an intelligence can receive,
generate, predict, remember, or act upon.

```text
signal:
  observable or generatable structure
  may be continuous or discrete
  does not require fixed human-defined meaning
```

A symbol is a sign-like unit that can stand for something.

```text
symbol:
  discrete or interpretable unit
  can refer, describe, command, classify, or explain
  may be learned rather than manually assigned
```

A token is a model-facing unit produced by a tokenizer or encoder.

```text
token:
  element of a sequence consumed by a training objective
  may come from text, image patches, audio frames, actions, or labels
```

Natural language is both signal and symbol. As model input it is a tokenizable
signal. As human language it also contains symbols that refer, describe,
command, classify, and explain.

Images and audio are usually less symbol-like at the raw input level, but they
can still produce symbol-like internal structure after learning.

## Representation Principle

The project should keep raw examples close to the source task.

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
```

These examples should become comparable only after a tokenizer, encoder, or
adapter has done modality-specific work.

```text
text example
  -> text tokenizer
  -> token ids

image example
  -> image adapter
  -> hidden sequence

image-choice example
  -> image adapter + label tokenizer
  -> hidden sequence / candidate continuation scores
```

This keeps the system simple. It avoids adding fields such as `channel`,
`role`, `kind`, or `target` just to make unrelated tasks appear uniform.

## Channel-Like Boundaries

A channel-like boundary is not an ontology. It is a practical route for a class
of data when that route helps implementation or evaluation.

```text
channel-like boundary:
  tokenizer / encoder selection
  decoder / renderer selection
  sequence boundary management
  dataset construction
  evaluation target selection
```

Such a boundary may reflect signal structure:

```text
text
image
audio
video
```

It may also reflect an interaction role when the experiment actually needs it:

```text
action
consequence
answer
```

But the boundary must earn its place. A name should not become a field or class
unless a tokenizer, adapter, dataset, evaluator, or training objective uses it.

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
inputs if the tokenizer treats them as text.

## Text As Reference Case

Text is the reference success case because it turns naturally into long,
learnable token sequences with abundant data.

```text
structured text
  -> tokenizer
  -> token ids
  -> next-token or continuation prediction
```

Other modalities should be judged by the same practical question:

```text
Can this raw example become a learnable token or hidden sequence?
```

They do not need to resemble natural language externally. Image patches, audio
frames, and action representations can all be valid if they create a sequence
that a shared model can learn from.

## Observation

`observation` is risky as a project abstraction because it is close to the
input side of signal itself.

```text
input signal ~= observation
```

This differs from concrete signal forms:

```text
text:
  signal form / tokenizer family

image:
  signal form / adapter family

observation:
  relation between a system and an incoming signal
```

Input is necessary, but an `Observation` class, field, or channel is not
automatically necessary. Natural language models receive input text without
usually modeling that input as an explicit `Observation` object.

Use an explicit observation boundary only when an implementation or evaluator
needs to distinguish received data from generated, predicted, remembered, or
action-like data.

## Action

`action` is also not a signal form in the same sense as text, image, audio, or
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
observation. Even then, the boundary should be added because the experiment
needs action-conditioned prediction or action generation, not because the
project wants a fixed ontology.

## Implementation Guidance

Prefer:

```text
task-specific raw examples
modality-specific tokenizers / encoders / adapters
TokenSequence for token-id training inputs
hidden sequences for shared Transformer cores
loss masks for separating context from supervised answer tokens
```

Avoid:

```text
generic raw-data envelopes
unused channel fields
schema fields that no tokenizer or evaluator consumes
handcrafted semantic taxonomies
making classification, prediction, memory, and action look identical too early
```

The shared middle of the system should be the model-facing representation:

```text
input adapter
  -> hidden sequence
  -> shared Transformer core
  -> output adapter / head
```

For text-only language modeling, the adapter may be a tokenizer and embedding
table. For image classification, it may be an image patch adapter plus a
classification head. For image-to-text or audio-to-text tasks, input and output
adapters can differ while the Transformer core remains shareable.

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
commonize the model-facing sequence representation
```
