# Shared Interface Boundary

Status: open.

## Issue

`shared_multimodal_model.py` and `shared_multimodal_checkpoint.py` still sit at
the top level because their responsibility is not clean enough to move into
`core/`.

They currently know concrete external interfaces and task heads, including
vision inputs, text routes, image classification, image-text choice, and
image-text answer outputs. Moving them into `core/` as-is would make `core/`
depend on concrete modalities and tasks.

Calling this area `multimodal/` is also premature. The real boundary may need
to cover shogi, grid, audio, video, and other external interfaces later, not
just image and text.

## Current Boundary

- `core/`: domain-agnostic sequence embedding and transformer utilities.
- External adapters: concrete input or state formats to embedding sequences.
- Task heads: hidden sequences to task-specific outputs.
- Tasks: compose adapters, shared core, and task heads.
- Task models: `tasks/<task>/model.py` wraps the interfaces, shared core, and
  prediction head for one concrete task.

`shared_multimodal_model.py` is a temporary multi-task model shell. New
task-specific model entry points should live under `tasks/<task>/model.py`
instead of adding more public routes to that shell.

## Acceptance Criteria

- `core/` does not import from `vision`, `text`, `shogi`, `grid`, or `tasks`.
- The concrete adapters and task heads inside `shared_multimodal_model.py` are
  identified before moving files.
- Any new package name matches the actual responsibility and is not a broad
  abstraction created ahead of need.
- Temporary shared model files are either split by responsibility or documented
  as an experimental shared-core route.
