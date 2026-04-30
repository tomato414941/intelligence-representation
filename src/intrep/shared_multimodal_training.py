from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.image_classification import ImageChoiceExample, image_label_tensors_from_examples
from intrep.language_modeling_training import (
    LanguageModelingTrainingDevice,
    language_model_batches,
    resolve_training_device,
)
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer


@dataclass(frozen=True)
class SharedMultimodalTrainingConfig:
    """Minimal shared-model training config.

    This currently covers only text language modeling plus image classification.
    Add new tasks only when their concrete training path exists.
    """

    text_context_length: int = 64
    image_patch_size: int = 4
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "tiny"
    device: LanguageModelingTrainingDevice = "cpu"
    tokenizer_vocab_size: int = 512


@dataclass(frozen=True)
class SharedMultimodalTrainingMetrics:
    text_token_count: int
    image_train_case_count: int
    text_initial_loss: float
    text_final_loss: float
    image_initial_loss: float
    image_final_loss: float
    image_train_accuracy: float
    max_steps: int
    model_preset: str


@dataclass(frozen=True)
class SharedMultimodalTrainingResult:
    metrics: SharedMultimodalTrainingMetrics
    model: SharedMultimodalModel
    tokenizer: TextTokenizer


def train_shared_multimodal_model(
    *,
    text_corpus: str,
    image_train_examples: list[ImageChoiceExample],
    config: SharedMultimodalTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> SharedMultimodalTrainingResult:
    training_config = config or SharedMultimodalTrainingConfig()
    _validate_config(training_config)
    if not text_corpus:
        raise ValueError("text_corpus must not be empty")
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)
    tokenizer = tokenizer_override or build_text_tokenizer(
        text_corpus,
        kind="byte-pair",
        vocab_size=training_config.tokenizer_vocab_size,
    )
    token_ids = tokenizer.encode(text_corpus)
    text_inputs, text_targets = language_model_batches(
        token_ids,
        context_length=training_config.text_context_length,
        batch_size=training_config.batch_size,
    )
    train_images, train_labels = image_label_tensors_from_examples(image_train_examples)
    preset = TRANSFORMER_CORE_PRESETS[training_config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=tokenizer.vocab_size,
        text_context_length=training_config.text_context_length,
        image_size=(int(train_images.shape[1]), int(train_images.shape[2])),
        patch_size=training_config.image_patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        num_classes=_class_count_from_examples(image_train_examples),
        channel_count=_channel_count_from_images(train_images),
    ).to(device)
    text_inputs = text_inputs.to(device)
    text_targets = text_targets.to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    text_initial_loss = _text_loss(model, loss_fn, text_inputs, text_targets)
    image_initial_loss = _image_loss(model, loss_fn, train_images, train_labels)

    for step in range(training_config.max_steps):
        optimizer.zero_grad(set_to_none=True)
        if step % 2 == 0:
            batch_index = (step // 2) % text_inputs.size(0)
            loss = _text_batch_loss(
                model,
                loss_fn,
                text_inputs[batch_index],
                text_targets[batch_index],
            )
        else:
            start = ((step // 2) * training_config.batch_size) % len(train_images)
            indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
            batch_images = train_images.index_select(0, indices)
            batch_labels = train_labels.index_select(0, indices)
            loss = loss_fn(model.image_logits(batch_images), batch_labels)
        loss.backward()
        optimizer.step()

    metrics = SharedMultimodalTrainingMetrics(
        text_token_count=len(token_ids),
        image_train_case_count=int(train_labels.numel()),
        text_initial_loss=text_initial_loss,
        text_final_loss=_text_loss(model, loss_fn, text_inputs, text_targets),
        image_initial_loss=image_initial_loss,
        image_final_loss=_image_loss(model, loss_fn, train_images, train_labels),
        image_train_accuracy=_image_accuracy(model, train_images, train_labels),
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )
    return SharedMultimodalTrainingResult(metrics=metrics, model=model, tokenizer=tokenizer)


def _text_batch_loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    logits = model.text_logits(inputs)
    return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def _text_loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        losses = [
            float(_text_batch_loss(model, loss_fn, batch_inputs, batch_targets).item())
            for batch_inputs, batch_targets in zip(inputs, targets)
        ]
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _image_loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = float(loss_fn(model.image_logits(images), labels).item())
    if was_training:
        model.train()
    return loss


def _image_accuracy(model: SharedMultimodalModel, images: torch.Tensor, labels: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model.image_logits(images).argmax(dim=1)
    if was_training:
        model.train()
    return float((predictions == labels).float().mean().item())


def _class_count_from_examples(examples: list[ImageChoiceExample]) -> int:
    if not examples:
        raise ValueError("examples must not be empty")
    choices = examples[0].choices
    for example in examples:
        if example.choices != choices:
            raise ValueError("all examples must use the same choices")
    return len(choices)


def _channel_count_from_images(images: torch.Tensor) -> int:
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[3])
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")


def _validate_config(config: SharedMultimodalTrainingConfig) -> None:
    if config.text_context_length <= 0:
        raise ValueError("text_context_length must be positive")
    if config.image_patch_size <= 0:
        raise ValueError("image_patch_size must be positive")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.model_preset not in TRANSFORMER_CORE_PRESETS:
        raise ValueError(f"unknown model preset: {config.model_preset}")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
