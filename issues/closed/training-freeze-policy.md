# Training Freeze Policy

Status: closed. Priority: low.

## Issue

The project can initialize a model from another checkpoint, but it cannot yet
control which model parts remain trainable after initialization.

Today, training generally optimizes all model parameters. That is too coarse for
experiments such as:

- freeze the shared core and train only a new output head
- freeze an input embedding module and train the core
- train only adapters or problem-specific heads
- compare full fine-tuning with partial fine-tuning

## Desired Direction

Use standard PyTorch mechanics:

- set `requires_grad = False` on frozen modules
- build optimizers from parameters where `requires_grad` is true
- keep `train()` / `eval()` separate from freezing

Start with a small helper such as `freeze_module(module)` only when a concrete
experiment needs it. Avoid introducing a broad training-plan abstraction before
the required cases are clear.

## Acceptance Criteria

- training code has a clear way to freeze selected model parts
- optimizer construction excludes frozen parameters
- freezing is not confused with `eval()` mode
- at least one concrete experiment or test exercises partial fine-tuning

## Resolution

Image/text choice and answer training now accept repeated `--freeze-module`
options. The training config records the selected modules, applies freezing
after checkpoint initialization and before optimizer construction, and reuses
the existing `build_adamw()` behavior that excludes `requires_grad=False`
parameters.

Tests cover partial fine-tuning by initializing from a compatible checkpoint,
freezing `core` and `image_input_layer`, training for one step, and verifying
that those modules remain equal to the initialization source.
