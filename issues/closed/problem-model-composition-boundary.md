# Problem Model Composition Boundary

Status: closed.

Resolution: problem models compose input embedding modules, shared core, and
output heads directly. The image/text shared shell was removed.

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

The former image/text shared shell was split into concrete problem models and
reusable input/output components.

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

Input embedding modules and output heads start near the problem model that needs
them. When two or more concrete problems need the same input or output boundary,
promote that module or head as a reusable component.

Sharing should happen at the module/head level, not by growing one universal
model class.

## Acceptance Criteria

- [x] problem models own concrete composition for their problem
- [x] the shared core interface is embedding sequence -> hidden state sequence
- [x] input embedding modules are shared when they are reusable concrete
  boundaries
- [x] output heads are shared when they are reusable concrete boundaries
- [x] shared input modules and output heads live near their interface
- [x] `ImageTextSharedModel` was split into smaller reusable pieces

## Non-Goals

- do not introduce a universal model class
- do not force every source into one input schema
- do not redesign mixed-schema datasets here
