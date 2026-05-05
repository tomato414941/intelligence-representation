from __future__ import annotations

import torch

from intrep.core.model_input import concatenate_input_embedding_sequences
from intrep.shared_multimodal_model import ImageTextSharedModel


class ImageTextAnswerModel(ImageTextSharedModel):
    """Task model for image-conditioned token prediction."""

    def token_logits(self, images: torch.Tensor, text_token_ids: torch.Tensor) -> torch.Tensor:
        if text_token_ids.ndim != 2:
            raise ValueError("text_token_ids must have shape [batch, sequence]")
        if text_token_ids.size(1) > self.text_context_length:
            raise ValueError("text_token_ids sequence length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        text_embeddings = self._text_embeddings(text_token_ids)
        combined = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
        return self.token_output(self.core(combined, causal=True))
