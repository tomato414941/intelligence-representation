from __future__ import annotations

import torch
from torch import nn

from intrep.core.model_input import concatenate_input_embedding_sequences
from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.text.input_layer import TextTokenInputLayer
from intrep.text.output_layer import TokenOutputHead
from intrep.vision.input_layer import ImagePatchInputLayer


class ImageTextAnswerModel(nn.Module):
    """Task model for image-conditioned token prediction."""

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
        self.image_input_layer = ImagePatchInputLayer(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            channel_count=channel_count,
        )
        self.text_input_layer = TextTokenInputLayer(
            vocab_size=vocab_size,
            context_length=text_context_length,
            embedding_dim=embedding_dim,
        )
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.token_output = TokenOutputHead(embedding_dim=embedding_dim, vocab_size=vocab_size)

    def token_logits(self, images: torch.Tensor, text_token_ids: torch.Tensor) -> torch.Tensor:
        if text_token_ids.ndim != 2:
            raise ValueError("text_token_ids must have shape [batch, sequence]")
        if text_token_ids.size(1) > self.text_context_length:
            raise ValueError("text_token_ids sequence length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        text_embeddings = self.text_input_layer(text_token_ids)
        combined = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
        return self.token_output(self.core(combined, causal=True))
