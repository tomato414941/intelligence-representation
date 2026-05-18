# Model Module Composition And Replacement

Status: open
Priority: high

## Problem

The project wants model construction to make input modules, shared cores, and
output heads explicit and replaceable.

Existing closed issues established two narrower policies:

- problem models compose input embedding modules, shared cores, and output heads
  directly
- checkpoint transfer can select reusable modules by name

That is not yet the same as a clean, first-class replacement boundary. In
practice, each problem model still decides its own constructor shape, checkpoint
metadata, module names, and compatibility rules. This makes changes such as the
shogi side-to-move-relative input harder to reason about than they should be.

## Desired Shape

Model assembly should make these parts explicit:

```text
input embedding module(s)
-> shared core
-> output head(s)
```

Each part should have a clear identity and responsibility:

- input modules own model-input encoding and conversion into input embedding
  sequences
- shared cores own representation transformation over embedding sequences
- output heads own task-specific interpretation of hidden states
- checkpoints record enough module identity to reject accidental incompatible
  restores
- transfer or initialization can intentionally reuse selected modules without
  relying on accidental state-dict key overlap

This does not require one universal model class. Concrete problem models can
remain concrete, but their composition boundary should be regular enough that a
new input module, core, or output head can be swapped deliberately.

## Current Pressure

Shogi input representation changes now affect:

- position input encoding
- candidate move encoding
- checkpoint compatibility
- training initialization
- inference and evaluation entry points

That is evidence that input modules need a cleaner identity and replacement
boundary.

## Progress

- Shogi policy/value now exposes separate model parts for position input,
  candidate move input, shared core, policy head, and value head.
- Shogi checkpoint metadata now records a model spec with position input,
  candidate move input, core, pooling, policy head, and value head identities in
  addition to the input encoding identity, so incompatible full-checkpoint
  restores are rejected deliberately.
- Shogi training config now selects a model by name (`shared_transformer` or
  `direct`) instead of using a `use_shared_core` implementation boolean.

## Non-Goals

- Do not force every modality into one raw input schema.
- Do not introduce a universal model class before there is a concrete need.
- Do not make arbitrary runtime hot-swapping the first goal.
- Do not weaken strict full-checkpoint restore semantics.

## Close Condition

- Define the project-level module composition boundary for input modules,
  shared cores, and output heads.
- Decide which identity each module type must expose in checkpoints.
- Update at least one concrete problem model, likely shogi policy/value, to use
  the chosen boundary.
- Keep full checkpoint restore strict and make intentional partial reuse
  explicit by module identity.
