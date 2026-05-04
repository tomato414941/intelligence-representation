from __future__ import annotations

import torch

from intrep.shared_multimodal_model import SharedMultimodalModel


class ImageTextChoiceModel(SharedMultimodalModel):
    """Task model for image-conditioned fixed choice scoring."""

    def choice_logits(
        self,
        images: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        choice_token_ids: torch.Tensor,
        choice_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.image_text_choice_logits(
            images,
            prompt_token_ids,
            choice_token_ids,
            choice_token_mask,
        )
