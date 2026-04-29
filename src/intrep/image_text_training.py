from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from intrep.causal_text_model import CausalTextModel
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer, image_label_tensors_from_examples
from intrep.image_text_scoring import TextTokenizer
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.token_scoring import next_token_loss


@dataclass(frozen=True)
class ImageTextTrainingConfig:
    max_steps: int = 20
    learning_rate: float = 0.003
    seed: int = 7


@dataclass(frozen=True)
class ImageTextTrainingResult:
    initial_loss: float
    final_loss: float
    steps: int
    case_count: int
    loss_history: tuple[float, ...]


def train_image_text_choices(
    *,
    examples: Sequence[ImageChoiceExample],
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    prompt: str,
    config: ImageTextTrainingConfig | None = None,
) -> ImageTextTrainingResult:
    if not examples:
        raise ValueError("examples must not be empty")
    training_config = config or ImageTextTrainingConfig()
    _validate_training_config(training_config)
    torch.manual_seed(training_config.seed)

    device = next(text_model.parameters()).device
    image_device = next(image_input_layer.parameters()).device
    if image_device != device:
        raise ValueError("image input layer and text model must be on the same device")
    if tokenizer.vocab_size != text_model.config.vocab_size:
        raise ValueError("tokenizer vocab size must match text model vocab size")

    images, labels = image_label_tensors_from_examples(list(examples))
    images = images.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(
        list(image_input_layer.parameters()) + list(text_model.parameters()),
        lr=training_config.learning_rate,
    )

    initial_loss = _mean_loss(
        examples=examples,
        images=images,
        labels=labels,
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    loss_history: list[float] = []
    image_input_layer.train()
    text_model.train()
    for step in range(training_config.max_steps):
        index = step % len(examples)
        optimizer.zero_grad(set_to_none=True)
        loss = _example_loss(
            example=examples[index],
            image=images[index],
            label=labels[index],
            image_input_layer=image_input_layer,
            text_model=text_model,
            tokenizer=tokenizer,
            prompt=prompt,
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    final_loss = _mean_loss(
        examples=examples,
        images=images,
        labels=labels,
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=tokenizer,
        prompt=prompt,
    )
    return ImageTextTrainingResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        steps=training_config.max_steps,
        case_count=len(examples),
        loss_history=tuple(loss_history),
    )


def _mean_loss(
    *,
    examples: Sequence[ImageChoiceExample],
    images: torch.Tensor,
    labels: torch.Tensor,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    prompt: str,
) -> float:
    image_was_training = image_input_layer.training
    text_was_training = text_model.training
    image_input_layer.eval()
    text_model.eval()
    try:
        with torch.no_grad():
            losses = [
                _example_loss(
                    example=example,
                    image=image,
                    label=label,
                    image_input_layer=image_input_layer,
                    text_model=text_model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                )
                for example, image, label in zip(examples, images, labels, strict=True)
            ]
            return float(torch.stack(losses).mean().item())
    finally:
        if image_was_training:
            image_input_layer.train()
        if text_was_training:
            text_model.train()


def _example_loss(
    *,
    example: ImageChoiceExample,
    image: torch.Tensor,
    label: torch.Tensor,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    prompt: str,
) -> torch.Tensor:
    answer_index = int(label.item())
    if answer_index != example.answer_index:
        raise ValueError("label tensor does not match example answer_index")
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(example.answer_text)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if not answer_ids:
        raise ValueError("answer text must encode to at least one token")

    image_embeddings = image_input_layer(image.unsqueeze(0))
    text_ids = torch.tensor(
        [prompt_ids + answer_ids],
        dtype=torch.long,
        device=image.device,
    )
    text_embeddings = text_model.embed_tokens(
        text_ids,
        position_offset=image_embeddings.size(1),
    )
    combined_embeddings = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
    hidden = text_model.encode_embeddings(combined_embeddings, causal=True)
    logits = text_model.token_logits(hidden)

    target_token_ids = torch.zeros(
        (1, combined_embeddings.size(1)),
        dtype=torch.long,
        device=image.device,
    )
    text_start = image_embeddings.size(1)
    text_end = text_start + text_ids.size(1)
    target_token_ids[:, text_start:text_end] = text_ids
    loss_mask = torch.zeros_like(target_token_ids, dtype=torch.bool)
    answer_start = text_start + len(prompt_ids)
    loss_mask[:, answer_start:text_end] = True
    return next_token_loss(logits, target_token_ids, loss_mask=loss_mask)


def _validate_training_config(config: ImageTextTrainingConfig) -> None:
    if config.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
