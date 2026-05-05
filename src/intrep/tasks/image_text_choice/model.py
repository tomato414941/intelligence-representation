from __future__ import annotations

import torch
from torch import nn

from intrep.core.model_input import concatenate_input_embedding_sequences
from intrep.shared_multimodal_model import ImageTextSharedModel


class ImageTextChoiceModel(ImageTextSharedModel):
    """Task model for image-conditioned fixed choice scoring."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.choice_score_head = nn.Linear(self.token_embedding.embedding_dim, 1)

    def choice_logits(
        self,
        images: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        choice_token_ids: torch.Tensor,
        choice_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_token_ids.ndim != 1:
            raise ValueError("prompt_token_ids must have shape [sequence]")
        if choice_token_ids.ndim != 2:
            raise ValueError("choice_token_ids must have shape [choice, sequence]")
        if choice_token_mask.shape != choice_token_ids.shape:
            raise ValueError("choice_token_mask must match choice_token_ids shape")
        if prompt_token_ids.size(0) + choice_token_ids.size(1) > self.text_context_length:
            raise ValueError("prompt plus choice token length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        prompt_embeddings = self._text_embeddings(prompt_token_ids.unsqueeze(0))
        choice_embeddings = self._text_embeddings(
            choice_token_ids,
            position_offset=prompt_token_ids.size(0),
        )
        batch_size = images.size(0)
        choice_count = choice_token_ids.size(0)
        expanded_images = image_embeddings[:, None, :, :].expand(-1, choice_count, -1, -1)
        expanded_prompts = prompt_embeddings[:, None, :, :].expand(batch_size, choice_count, -1, -1)
        expanded_choices = choice_embeddings[None, :, :, :].expand(batch_size, -1, -1, -1)
        image_sequence_length = image_embeddings.size(1)
        expanded_image_rows = expanded_images.reshape(batch_size * choice_count, image_sequence_length, -1)
        expanded_choice_rows = expanded_choices.reshape(
            batch_size * choice_count,
            choice_token_ids.size(1),
            -1,
        )
        if prompt_token_ids.numel() == 0:
            combined = concatenate_input_embedding_sequences(expanded_image_rows, expanded_choice_rows)
        else:
            combined = concatenate_input_embedding_sequences(
                expanded_image_rows,
                expanded_prompts.reshape(batch_size * choice_count, prompt_token_ids.size(0), -1),
                expanded_choice_rows,
            )
        hidden = self.core(combined, causal=False)
        choice_start = image_sequence_length + prompt_token_ids.size(0)
        choice_hidden = hidden[:, choice_start:, :]
        expanded_mask = choice_token_mask[None, :, :].expand(batch_size, -1, -1)
        mask = expanded_mask.reshape(batch_size * choice_count, -1).unsqueeze(-1).to(hidden.dtype)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (choice_hidden * mask).sum(dim=1) / token_counts
        return self.choice_score_head(pooled).reshape(batch_size, choice_count)
