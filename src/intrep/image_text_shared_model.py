from __future__ import annotations

import torch
from torch import nn

from intrep.text.causal_model import TokenOutputHead
from intrep.vision.input_layer import ImagePatchInputLayer
from intrep.core.transformer_core import SharedTransformerCore


class ImageTextSharedModel(nn.Module):
    """Shared image/text input shell used by problem-specific models."""

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

    def text_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.text_context_length:
            raise ValueError("token_ids sequence length must not exceed text_context_length")
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.text_position_embedding(positions)
        return self.token_output(self.core(embeddings, causal=True))

    def _text_embeddings(self, token_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            position_offset,
            position_offset + token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        return self.token_embedding(token_ids) + self.text_position_embedding(positions)
