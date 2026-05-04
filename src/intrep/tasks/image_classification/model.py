from __future__ import annotations

import torch

from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageClassificationModel(SharedMultimodalModel):
    """Task model for image-conditioned fixed class prediction."""

    def class_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_classification_logits(images)
