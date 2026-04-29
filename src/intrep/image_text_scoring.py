from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from intrep.causal_text_model import CausalTextModel
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer, image_label_tensors_from_examples
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.token_scoring import next_token_loss


class TextTokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]:
        ...


@dataclass(frozen=True)
class ImageTextChoiceMetrics:
    case_count: int
    accuracy: float
    predicted_indices: tuple[int, ...]
    losses: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "accuracy": self.accuracy,
            "predicted_indices": list(self.predicted_indices),
            "losses": [list(row) for row in self.losses],
        }


def evaluate_image_text_choices(
    *,
    examples: Sequence[ImageChoiceExample],
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    prompt: str,
) -> ImageTextChoiceMetrics:
    if not examples:
        raise ValueError("examples must not be empty")
    images, labels = image_label_tensors_from_examples(list(examples))
    predicted_indices: list[int] = []
    loss_rows: list[tuple[float, ...]] = []
    for image, example in zip(images, examples, strict=True):
        losses = score_image_text_candidates(
            image_input_layer=image_input_layer,
            text_model=text_model,
            tokenizer=tokenizer,
            image=image,
            prompt=prompt,
            candidates=example.choices,
        )
        loss_rows.append(tuple(losses))
        predicted_indices.append(min(range(len(losses)), key=losses.__getitem__))

    correct_count = sum(
        int(predicted_index == int(label.item()))
        for predicted_index, label in zip(predicted_indices, labels, strict=True)
    )
    return ImageTextChoiceMetrics(
        case_count=len(examples),
        accuracy=correct_count / len(examples),
        predicted_indices=tuple(predicted_indices),
        losses=tuple(loss_rows),
    )


def score_image_text_candidates(
    *,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    image: torch.Tensor,
    prompt: str,
    candidates: Sequence[str],
) -> list[float]:
    candidate_losses: list[float] = []
    for candidate in candidates:
        candidate_losses.append(
            _score_image_text_candidate(
                image_input_layer=image_input_layer,
                text_model=text_model,
                tokenizer=tokenizer,
                image=image,
                prompt=prompt,
                candidate=candidate,
            )
        )
    if not candidate_losses:
        raise ValueError("candidates must not be empty")
    return candidate_losses


def choose_image_text_candidate(
    *,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    image: torch.Tensor,
    prompt: str,
    candidates: Sequence[str],
) -> int:
    losses = score_image_text_candidates(
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=tokenizer,
        image=image,
        prompt=prompt,
        candidates=candidates,
    )
    return min(range(len(losses)), key=losses.__getitem__)


def _score_image_text_candidate(
    *,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    image: torch.Tensor,
    prompt: str,
    candidate: str,
) -> float:
    prompt_ids = tokenizer.encode(prompt)
    candidate_ids = tokenizer.encode(candidate)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if not candidate_ids:
        raise ValueError("candidate must encode to at least one token")
    if tokenizer.vocab_size != text_model.config.vocab_size:
        raise ValueError("tokenizer vocab size must match text model vocab size")

    text_was_training = text_model.training
    image_was_training = image_input_layer.training
    text_model.eval()
    image_input_layer.eval()
    try:
        with torch.no_grad():
            device = next(text_model.parameters()).device
            image_device = next(image_input_layer.parameters()).device
            if image_device != device:
                raise ValueError("image input layer and text model must be on the same device")
            image_batch = _image_batch(image).to(device)
            image_embeddings = image_input_layer(image_batch)
            text_ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
            text_embeddings = text_model.embed_tokens(text_ids)
            combined_embeddings = concatenate_input_embedding_sequences(
                image_embeddings,
                text_embeddings,
            )
            hidden = text_model.encode_embeddings(combined_embeddings, causal=True)
            logits = text_model.token_logits(hidden)

            target_token_ids = torch.zeros(
                (1, combined_embeddings.size(1)),
                dtype=torch.long,
                device=device,
            )
            text_start = image_embeddings.size(1)
            text_end = text_start + text_ids.size(1)
            target_token_ids[:, text_start:text_end] = text_ids
            loss_mask = torch.zeros_like(target_token_ids, dtype=torch.bool)
            candidate_start = text_start + len(prompt_ids)
            loss_mask[:, candidate_start:text_end] = True
            return float(next_token_loss(logits, target_token_ids, loss_mask=loss_mask).item())
    finally:
        if text_was_training:
            text_model.train()
        if image_was_training:
            image_input_layer.train()


def _image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3 and image.size(0) == 1:
        return image
    raise ValueError("image must have shape [height, width] or [1, height, width]")
