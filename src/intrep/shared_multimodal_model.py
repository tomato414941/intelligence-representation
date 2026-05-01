from __future__ import annotations

import torch
from torch import nn

from intrep.causal_text_model import TokenOutputHead
from intrep.image_classification import ClassificationHead, ImagePatchInputLayer
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.transformer_core import SharedTransformerCore


class SharedMultimodalModel(nn.Module):
    """Shared model shell for text-token and image-patch inputs.

    This is not a universal multimodal model yet. It currently exposes text
    language-model logits, image classification logits, and image-text
    candidate scores over one shared Transformer core.
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
        num_classes: int,
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
        self.classification_head = ClassificationHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )
        self.candidate_score_head = nn.Linear(embedding_dim, 1)

    def text_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.text_context_length:
            raise ValueError("token_ids sequence length must not exceed text_context_length")
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.text_position_embedding(positions)
        return self.token_output(self.core(embeddings, causal=True))

    def image_logits(self, images: torch.Tensor) -> torch.Tensor:
        embeddings = self.image_input_layer(images)
        return self.classification_head(self.core(embeddings, causal=False))

    def image_text_fusion_candidate_logits(
        self,
        images: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_token_ids.ndim != 2:
            raise ValueError("candidate_token_ids must have shape [candidate, sequence]")
        if candidate_token_mask.shape != candidate_token_ids.shape:
            raise ValueError("candidate_token_mask must match candidate_token_ids shape")
        if candidate_token_ids.size(1) > self.text_context_length:
            raise ValueError("candidate token length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        candidate_embeddings = self._candidate_text_embeddings(candidate_token_ids)
        batch_size = images.size(0)
        candidate_count = candidate_token_ids.size(0)
        expanded_images = image_embeddings[:, None, :, :].expand(-1, candidate_count, -1, -1)
        expanded_candidates = candidate_embeddings[None, :, :, :].expand(batch_size, -1, -1, -1)
        image_sequence_length = image_embeddings.size(1)
        combined = concatenate_input_embedding_sequences(
            expanded_images.reshape(batch_size * candidate_count, image_sequence_length, -1),
            expanded_candidates.reshape(batch_size * candidate_count, candidate_token_ids.size(1), -1),
        )
        hidden = self.core(combined, causal=False)
        candidate_hidden = hidden[:, image_sequence_length:, :]
        expanded_mask = candidate_token_mask[None, :, :].expand(batch_size, -1, -1)
        mask = expanded_mask.reshape(batch_size * candidate_count, -1).unsqueeze(-1).to(hidden.dtype)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        pooled = (candidate_hidden * mask).sum(dim=1) / token_counts
        return self.candidate_score_head(pooled).reshape(batch_size, candidate_count)

    def _candidate_text_embeddings(self, candidate_token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(candidate_token_ids.size(1), device=candidate_token_ids.device).unsqueeze(0)
        return self.token_embedding(candidate_token_ids) + self.text_position_embedding(positions)
