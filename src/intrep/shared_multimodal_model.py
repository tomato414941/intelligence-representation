from __future__ import annotations

import torch
from torch import nn

from intrep.causal_text_model import TokenOutputHead
from intrep.image_classification import ImagePatchInputLayer
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.transformer_core import SharedTransformerCore


class SharedMultimodalModel(nn.Module):
    """Shared model shell for text-token and image-patch inputs.

    This is not a universal multimodal model yet. It currently exposes text
    language-model logits and image-text candidate scores over one shared
    Transformer core.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        text_context_length: int,
        image_size: tuple[int, int],
        patch_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        channel_count: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.text_context_length = text_context_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.text_position_embedding = nn.Embedding(text_context_length, embedding_dim)
        self.image_input_layer = ImagePatchInputLayer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            channel_count=channel_count,
        )
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.token_output = TokenOutputHead(embedding_dim=embedding_dim, vocab_size=vocab_size)
        self.candidate_score_head = nn.Linear(embedding_dim, 1)

    def text_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.text_context_length:
            raise ValueError("token_ids sequence length must not exceed text_context_length")
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.text_position_embedding(positions)
        return self.token_output(self.core(embeddings, causal=True))

    def image_text_fusion_candidate_logits(
        self,
        images: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_token_ids.ndim != 1:
            raise ValueError("prompt_token_ids must have shape [sequence]")
        if candidate_token_ids.ndim != 2:
            raise ValueError("candidate_token_ids must have shape [candidate, sequence]")
        if candidate_token_mask.shape != candidate_token_ids.shape:
            raise ValueError("candidate_token_mask must match candidate_token_ids shape")
        if prompt_token_ids.size(0) + candidate_token_ids.size(1) > self.text_context_length:
            raise ValueError("prompt plus candidate token length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        prompt_embeddings = self._text_embeddings(prompt_token_ids.unsqueeze(0))
        candidate_embeddings = self._text_embeddings(
            candidate_token_ids,
            position_offset=prompt_token_ids.size(0),
        )
        batch_size = images.size(0)
        candidate_count = candidate_token_ids.size(0)
        expanded_images = image_embeddings[:, None, :, :].expand(-1, candidate_count, -1, -1)
        expanded_prompts = prompt_embeddings[:, None, :, :].expand(batch_size, candidate_count, -1, -1)
        expanded_candidates = candidate_embeddings[None, :, :, :].expand(batch_size, -1, -1, -1)
        image_sequence_length = image_embeddings.size(1)
        expanded_image_rows = expanded_images.reshape(batch_size * candidate_count, image_sequence_length, -1)
        expanded_candidate_rows = expanded_candidates.reshape(
            batch_size * candidate_count,
            candidate_token_ids.size(1),
            -1,
        )
        if prompt_token_ids.numel() == 0:
            combined = concatenate_input_embedding_sequences(expanded_image_rows, expanded_candidate_rows)
        else:
            combined = concatenate_input_embedding_sequences(
                expanded_image_rows,
                expanded_prompts.reshape(batch_size * candidate_count, prompt_token_ids.size(0), -1),
                expanded_candidate_rows,
            )
        hidden = self.core(combined, causal=False)
        candidate_start = image_sequence_length + prompt_token_ids.size(0)
        candidate_hidden = hidden[:, candidate_start:, :]
        expanded_mask = candidate_token_mask[None, :, :].expand(batch_size, -1, -1)
        mask = expanded_mask.reshape(batch_size * candidate_count, -1).unsqueeze(-1).to(hidden.dtype)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (candidate_hidden * mask).sum(dim=1) / token_counts
        return self.candidate_score_head(pooled).reshape(batch_size, candidate_count)

    def _text_embeddings(self, token_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            position_offset,
            position_offset + token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        return self.token_embedding(token_ids) + self.text_position_embedding(positions)
