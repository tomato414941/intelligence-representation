# Shared Interface Boundary

Status: open. Priority: low.

## Issue

`image_text_shared_model.py` sits at the top level because its responsibility is
not clean enough to move into `core/`, but it is also not a single problem
model.

It knows concrete external interfaces:

- image patch input
- text token and position embeddings
- token output
- shared Transformer core wiring

It is currently shared by:

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

`image_text_shared_model.py` is a temporary image/text shared model shell. New
problem-specific model entry points should live under `problems/<problem>/model.py`.
Do not add more public routes to `ImageTextSharedModel` unless at least two
current image/text problems need the same route.

## Acceptance Criteria

- `core/` does not import from `vision`, `text`, `shogi`, `grid`, or `problems`.
- The concrete adapters and reusable routes inside `image_text_shared_model.py`
  are identified before moving files.
- Any new package name matches the actual responsibility and is not a broad
  abstraction created ahead of need.
- `image_text_shared_model.py` is either kept as an explicit image/text shared
  shell, split by responsibility, or moved only after a clearer package boundary
  exists.
