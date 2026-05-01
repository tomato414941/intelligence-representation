from __future__ import annotations

import torch
from torch import nn

from intrep.causal_text_model import TokenOutputHead
from intrep.image_input_layer import ImagePatchInputLayer
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.transformer_core import SharedTransformerCore


class ClassificationHead(nn.Module):
    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        pooled = hidden.mean(dim=1)
        return self.output(pooled)


class SharedMultimodalModel(nn.Module):
    """Shared model shell for text-token and image-patch inputs.

    It exposes text-token logits, image-conditioned token logits, and
    image-text choice scores over one shared Transformer core.
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
        num_classes: int | None = None,
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
        self.choice_score_head = nn.Linear(embedding_dim, 1)
        self.classification_head = (
            ClassificationHead(embedding_dim=embedding_dim, num_classes=num_classes)
            if num_classes is not None
            else None
        )

    def text_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        if token_ids.size(1) > self.text_context_length:
            raise ValueError("token_ids sequence length must not exceed text_context_length")
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(token_ids) + self.text_position_embedding(positions)
        return self.token_output(self.core(embeddings, causal=True))

    def image_text_token_logits(self, images: torch.Tensor, text_token_ids: torch.Tensor) -> torch.Tensor:
        if text_token_ids.ndim != 2:
            raise ValueError("text_token_ids must have shape [batch, sequence]")
        if text_token_ids.size(1) > self.text_context_length:
            raise ValueError("text_token_ids sequence length must not exceed text_context_length")
        image_embeddings = self.image_input_layer(images)
        text_embeddings = self._text_embeddings(text_token_ids)
        combined = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
        return self.token_output(self.core(combined, causal=True))

    def image_classification_logits(self, images: torch.Tensor) -> torch.Tensor:
        if self.classification_head is None:
            raise ValueError("model was created without a classification head")
        return self.classify_embeddings(self.encode_images(images))

    def embed_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_input_layer(images)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.core(self.embed_images(images), causal=False)

    def classify_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.classification_head is None:
            raise ValueError("model was created without a classification head")
        return self.classification_head(embeddings)

    def image_text_choice_logits(
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

    def _text_embeddings(self, token_ids: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(
            position_offset,
            position_offset + token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        return self.token_embedding(token_ids) + self.text_position_embedding(positions)
