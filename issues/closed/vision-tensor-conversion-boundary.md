# Vision tensor conversion boundary

Status: closed.

Resolution: solved on 2026-06-11 by splitting the conversion. Image
normalization and shape logic stay in `sources/vision/io.py` as plain numpy,
removing the torch dependency from the vision source package; each image
problem wraps the array into a tensor with a one-line local helper at its
dataset edge. No new shared package was introduced.

## Issue

`src/intrep/domains/vision/training_data.py` converts image files into `torch.Tensor`.

That makes the vision domain package depend on the training/runtime tensor
representation. The vision domain should describe image data and image file
formats; tensor conversion belongs closer to the problem training path or a
representation input adapter.

## Desired Shape

- `domains/vision` keeps image-format and image-array logic.
- Torch tensor conversion is owned by the image problem training code or a
  representation input utility.
- Image classification, image-text-choice, and image-text-answer training paths
  continue to share the conversion code without making `domains/vision` depend
  on torch.

## Scope

- Move `image_tensor_from_path`.
- Move `channel_count_from_image_shape` if it remains tied to tensor training
  setup.
- Update tests to target the new owner.
