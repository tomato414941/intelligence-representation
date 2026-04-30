from __future__ import annotations

"""Temporary validation path for shared-core multitask training.

This module intentionally covers only text language modeling plus image
classification. Once the shared-core approach is proven enough, replace this
task-specific path with the main multitask training path or delete it.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from intrep.image_classification import ImageChoiceExample, image_label_tensors_from_examples
from intrep.language_modeling_training import (
    LanguageModelingDataset,
    LanguageModelingTrainingDevice,
    resolve_training_device,
)
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer


@dataclass(frozen=True)
class TextImageSharedCoreTrainingConfig:
    text_context_length: int = 64
    patch_size: int = 4
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "tiny"
    device: LanguageModelingTrainingDevice = "cpu"
    tokenizer_vocab_size: int = 512


@dataclass(frozen=True)
class TextImageSharedCoreMetrics:
    text_token_count: int
    image_train_case_count: int
    image_eval_case_count: int
    text_initial_loss: float
    text_final_loss: float
    image_initial_loss: float
    image_final_loss: float
    image_train_accuracy: float
    image_eval_accuracy: float | None
    max_steps: int
    model_preset: str
    shared_core: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "text_token_count": self.text_token_count,
            "image_train_case_count": self.image_train_case_count,
            "image_eval_case_count": self.image_eval_case_count,
            "text_initial_loss": self.text_initial_loss,
            "text_final_loss": self.text_final_loss,
            "image_initial_loss": self.image_initial_loss,
            "image_final_loss": self.image_final_loss,
            "image_train_accuracy": self.image_train_accuracy,
            "image_eval_accuracy": self.image_eval_accuracy,
            "max_steps": self.max_steps,
            "model_preset": self.model_preset,
            "shared_core": self.shared_core,
        }


@dataclass(frozen=True)
class TextImageSharedCoreTrainingResult:
    metrics: TextImageSharedCoreMetrics
    model: SharedMultimodalModel
    tokenizer: TextTokenizer


def train_text_image_shared_core_with_result(
    *,
    text_corpus: str,
    image_train_examples: list[ImageChoiceExample],
    image_eval_examples: list[ImageChoiceExample] | None = None,
    config: TextImageSharedCoreTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> TextImageSharedCoreTrainingResult:
    training_config = config or TextImageSharedCoreTrainingConfig()
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
    text_dataset = LanguageModelingDataset(
        token_ids,
        context_length=training_config.text_context_length,
    )
    text_loader = _text_data_loader(
        text_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=True,
    )
    text_eval_loader = _text_data_loader(
        text_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=False,
    )

    train_images, train_labels = image_label_tensors_from_examples(image_train_examples)
    eval_images: torch.Tensor | None = None
    eval_labels: torch.Tensor | None = None
    if image_eval_examples is not None:
        eval_images, eval_labels = image_label_tensors_from_examples(image_eval_examples)
        if tuple(eval_images.shape[1:]) != tuple(train_images.shape[1:]):
            raise ValueError("eval images must have the same shape as train images")
        _validate_choice_set(image_train_examples, image_eval_examples)

    preset = TRANSFORMER_CORE_PRESETS[training_config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=tokenizer.vocab_size,
        text_context_length=training_config.text_context_length,
        image_size=(int(train_images.shape[1]), int(train_images.shape[2])),
        patch_size=training_config.patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        num_classes=_class_count_from_examples(image_train_examples),
        channel_count=_channel_count_from_images(train_images),
    ).to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    if eval_images is not None and eval_labels is not None:
        eval_images = eval_images.to(device)
        eval_labels = eval_labels.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    text_initial_loss = _text_loss(model, loss_fn, text_eval_loader, device)
    image_initial_loss = _image_loss(model, loss_fn, train_images, train_labels)

    text_iterator = iter(text_loader)
    for step in range(training_config.max_steps):
        optimizer.zero_grad(set_to_none=True)
        if step % 2 == 0:
            try:
                batch_inputs, batch_targets = next(text_iterator)
            except StopIteration:
                text_iterator = iter(text_loader)
                batch_inputs, batch_targets = next(text_iterator)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model.text_logits(batch_inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
        else:
            start = ((step // 2) * training_config.batch_size) % len(train_images)
            indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
            batch_images = train_images.index_select(0, indices)
            batch_labels = train_labels.index_select(0, indices)
            loss = loss_fn(model.image_logits(batch_images), batch_labels)
        loss.backward()
        optimizer.step()

    text_final_loss = _text_loss(model, loss_fn, text_eval_loader, device)
    image_final_loss = _image_loss(model, loss_fn, train_images, train_labels)
    image_train_accuracy = _image_accuracy(model, train_images, train_labels)
    image_eval_accuracy = None
    image_eval_case_count = 0
    if eval_images is not None and eval_labels is not None:
        image_eval_accuracy = _image_accuracy(model, eval_images, eval_labels)
        image_eval_case_count = int(eval_labels.numel())
    metrics = TextImageSharedCoreMetrics(
        text_token_count=len(token_ids),
        image_train_case_count=int(train_labels.numel()),
        image_eval_case_count=image_eval_case_count,
        text_initial_loss=text_initial_loss,
        text_final_loss=text_final_loss,
        image_initial_loss=image_initial_loss,
        image_final_loss=image_final_loss,
        image_train_accuracy=image_train_accuracy,
        image_eval_accuracy=image_eval_accuracy,
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )
    return TextImageSharedCoreTrainingResult(metrics=metrics, model=model, tokenizer=tokenizer)


def _text_loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    data_loader,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            logits = model.text_logits(batch_inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch_targets.reshape(-1))
            total += float(loss.item())
            batch_count += 1
    if was_training:
        model.train()
    return total / batch_count


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


def _image_accuracy(
    model: SharedMultimodalModel,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model.image_logits(images).argmax(dim=1)
    if was_training:
        model.train()
    return float((predictions == labels).float().mean().item())


def _validate_config(config: TextImageSharedCoreTrainingConfig) -> None:
    if config.text_context_length <= 0:
        raise ValueError("text_context_length must be positive")
    if config.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if config.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
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


def _text_data_loader(
    dataset: LanguageModelingDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _class_count_from_examples(examples: list[ImageChoiceExample]) -> int:
    if not examples:
        raise ValueError("examples must not be empty")
    choices = examples[0].choices
    for example in examples:
        if example.choices != choices:
            raise ValueError("all examples must use the same choices")
    return len(choices)


def _validate_choice_set(
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample],
) -> None:
    train_choices = train_examples[0].choices
    for example in eval_examples:
        if example.choices != train_choices:
            raise ValueError("eval examples must use the same choices as train examples")


def _channel_count_from_images(images: torch.Tensor) -> int:
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[3])
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")
