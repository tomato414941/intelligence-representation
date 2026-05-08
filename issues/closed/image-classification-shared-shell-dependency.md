# Image Classification Shared Shell Dependency

Status: closed.

## Issue

`ImageClassificationModel` currently inherits from `ImageTextSharedModel`.

Image classification only needs:

```text
image input
  -> shared core
  -> classification head
```

But inheriting from `ImageTextSharedModel` also gives it text/token routes that
image classification does not need:

- token embedding
- text position embedding
- token output head
- `text_logits()`
- `_text_embeddings()`

## Why It Matters

This is not a runtime bug, but it makes the shared unit too large.

The project direction is to share reusable modules and heads where useful, not
to grow one broad shared model shell that every problem inherits from.

## Candidate Fix

Make `ImageClassificationModel` compose only the pieces it needs:

- `ImagePatchInputLayer`
- `SharedTransformerCore`
- `ClassificationHead`

Keep `ImageTextSharedModel` for image/text problems unless a separate concrete
reason appears to split it further.

## Acceptance Criteria

- [x] `ImageClassificationModel` no longer inherits from `ImageTextSharedModel`.
- [x] Image classification still uses `ImagePatchInputLayer`, `SharedTransformerCore`,
  and `ClassificationHead`.
- [x] Existing image classification tests pass.

## Resolution

`ImageClassificationModel` now composes only the pieces it needs:

- `ImagePatchInputLayer`
- `SharedTransformerCore`
- `ClassificationHead`

It no longer inherits the image/text shell or exposes text/token routes.

## Non-Goals

- do not split `ImageTextSharedModel` entirely
- do not change image-text choice or image-text answer models
- do not introduce a generic model-composition framework
