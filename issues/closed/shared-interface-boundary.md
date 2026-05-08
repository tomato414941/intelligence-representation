# Shared Interface Boundary

Status: closed.

Resolution: `image_text_shared_model.py` was removed. Image/text problem models
now compose image input, text input, shared core, and output heads directly.

## Issue

`image_text_shared_model.py` sat at the top level because its responsibility was
not clean enough to move into `core/`, but it is also not a single problem
model.

It knows concrete external interfaces:

- image patch input
- text token and position embeddings
- token output
- shared Transformer core wiring

It was shared by:

- `problems/image_text_choice/model.py`
- `problems/image_text_answer/model.py`

`problems/image_classification/model.py` was split out of this shell and now
composes `ImagePatchInputLayer`, `SharedTransformerCore`, and
`ClassificationHead` directly.

Moving it into `core/` as-is would make `core/` depend on concrete image/text
interfaces.

Do not assume a future `multimodal/` package. If shared cross-interface code
appears, name the package by responsibility, such as routing, fusion, alignment,
or interface composition, not by the broad fact that multiple modalities are
involved.

## Current Boundary

- `core/`: domain-agnostic sequence embedding and transformer utilities.
- External adapters: concrete input or state formats to embedding sequences.
- Problem heads: hidden sequences to problem-specific outputs.
- Problem models: compose adapters, shared core, and problem heads.
- `problems/<problem>/model.py` wraps the interfaces, shared core, and
  prediction head for one concrete task.

Problem-specific model entry points live under `problems/<problem>/model.py`.
Shared behavior should be extracted as named input modules, output heads, or core
interfaces instead of a broad image/text shell.

## Acceptance Criteria

- `core/` does not import from `vision`, `text`, `shogi`, `grid`, or `problems`.
- [x] `core/` does not import from `vision`, `text`, `shogi`, `grid`, or
  `problems`.
- [x] The concrete adapters and reusable routes inside `image_text_shared_model.py`
  were split by responsibility.
- [x] No broad `multimodal/` package was introduced.
- [x] `image_text_shared_model.py` was removed.
