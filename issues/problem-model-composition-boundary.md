# Problem Model Composition Boundary

Status: open.

## Issue

The project needs a clear boundary for how problem models are composed around a
shared core.

The goal is not to grow one universal model class that owns every input and
output route. The likely direction is that each problem model owns the concrete
parts it needs while using a common shared-core interface.

## Current Shape

Problem models currently mix these pieces in slightly different ways:

```text
input embedding module
  -> shared core
  -> output head / decoder
```

Examples:

- image classification: image patch input, shared core, classification head
- image-text answer: image input, text input, shared core, token output head
- image-text choice: image input, text input, shared core, choice scoring head
- shogi policy/value: shogi position input, shared core, policy/value heads
- grid step prediction: grid input, shared core, prediction heads

`ImageTextSharedModel` is a useful image/text shell, but it should not become
the place where every future interface is added.

## Why It Matters

If the project grows one broad shared model class, concrete image/text choices
can become the hidden default for shogi, grid, audio, video, or future
interfaces.

If every problem model is completely independent, shared-core transfer and
mixed-problem learning become harder to reason about.

The important boundary is the shared-core connection, not a universal raw input
schema or a universal model class.

## Current Policy

Keep problem models concrete.

Share the core interface where useful:

```text
input embedding sequence -> shared core -> hidden state sequence
```

Leave input embedding modules and output heads problem-specific unless a second
concrete problem needs the same code.

## Acceptance Criteria

This issue can close when the project documents or implements a stable rule for:

- what a problem model owns
- what the shared core interface requires
- when an input embedding module should be shared
- when an output head should be shared
- whether `ImageTextSharedModel` remains a temporary image/text shell or is split
  into smaller reusable pieces

## Non-Goals

- do not introduce a universal model class
- do not force every source into one input schema
- do not redesign mixed-schema datasets here
