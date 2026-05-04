from __future__ import annotations

import torch

from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageTextAnswerModel(SharedMultimodalModel):
    """Task model for image-conditioned token prediction."""

    def token_logits(self, images: torch.Tensor, text_token_ids: torch.Tensor) -> torch.Tensor:
        return self.image_text_token_logits(images, text_token_ids)
