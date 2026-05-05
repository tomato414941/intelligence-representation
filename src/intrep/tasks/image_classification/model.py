from __future__ import annotations

import torch
from torch import nn

from intrep.core.transformer_core import SharedTransformerCore
from intrep.image_text_shared_model import ImageTextSharedModel
from intrep.vision.input_layer import ImagePatchInputLayer


class ClassificationHead(nn.Module):
    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        pooled = hidden.mean(dim=1)
        return self.output(pooled)


class ImageClassificationModel(ImageTextSharedModel):
    """Task model for image-conditioned fixed class prediction."""

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
        super().__init__(
            vocab_size=vocab_size,
            text_context_length=text_context_length,
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            channel_count=channel_count,
            dropout=dropout,
        )
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
        self.classification_head = ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes)

    def class_logits(self, images: torch.Tensor) -> torch.Tensor:
        return self.classify_embeddings(self.encode_images(images))

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_input_layer(images)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.core(self.embed_images(images), causal=False)

    def classify_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classification_head(embeddings)
