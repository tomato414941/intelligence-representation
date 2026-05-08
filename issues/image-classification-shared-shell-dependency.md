# Image Classification Shared Shell Dependency

Status: open.

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

- `ImageClassificationModel` no longer inherits from `ImageTextSharedModel`.
- Image classification still uses `ImagePatchInputLayer`, `SharedTransformerCore`,
  and `ClassificationHead`.
- Existing image classification tests pass.

## Non-Goals

- do not split `ImageTextSharedModel` entirely
- do not change image-text choice or image-text answer models
- do not introduce a generic model-composition framework
