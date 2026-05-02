from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from intrep.image_text_choice_examples import ImageTextChoiceExample
from intrep.image_training_data import (
    channel_count_from_image_shape,
    image_tensor_from_path,
    seeded_data_loader,
)
from intrep.language_modeling_training import (
    LanguageModelingDataset,
    LanguageModelingTrainingDevice,
    resolve_training_device,
)
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.shared_state_loading import load_compatible_shared_state
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer
from intrep.training_utils import LearningRateSchedule, build_adamw, build_lr_scheduler, clip_gradients


@dataclass(frozen=True)
class ImageTextChoiceTrainingConfig:
    text_context_length: int = 16
    image_patch_size: int = 4
    max_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    seed: int = 7
    model_preset: str = "tiny"
    device: LanguageModelingTrainingDevice = "cpu"
    tokenizer_vocab_size: int = 512


@dataclass(frozen=True)
class ImageTextChoiceMetrics:
    train_case_count: int
    eval_case_count: int
    train_initial_loss: float
    train_final_loss: float
    train_accuracy: float
    eval_accuracy: float | None
    text_initial_loss: float | None
    text_final_loss: float | None
    max_steps: int
    model_preset: str


@dataclass(frozen=True)
class ImageTextChoiceTrainingResult:
    metrics: ImageTextChoiceMetrics
    model: SharedMultimodalModel
    tokenizer: TextTokenizer
    config: ImageTextChoiceTrainingConfig
    image_shape: tuple[int, ...]


@dataclass(frozen=True)
class ImageTextChoiceEvalMetrics:
    case_count: int
    loss: float
    accuracy: float


class ImageTextChoiceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[ImageTextChoiceExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        _choices_from_examples(examples)
        self.examples = tuple(examples)
        self.image_shape = tuple(int(value) for value in image_tensor_from_path(examples[0].image_path).shape)
        self.channel_count = channel_count_from_image_shape(self.image_shape)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        image = image_tensor_from_path(example.image_path)
        if tuple(image.shape) != self.image_shape:
            raise ValueError("all images must have the same shape")
        return image, torch.tensor(example.answer_index, dtype=torch.long)


def train_image_text_choice_model(
    *,
    train_examples: list[ImageTextChoiceExample],
    eval_examples: list[ImageTextChoiceExample] | None = None,
    tokenizer_corpus: str = "",
    language_modeling_corpus: str | None = None,
    prompt: str = "",
    additional_prompts: tuple[str, ...] = (),
    config: ImageTextChoiceTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
    initial_model_state_dict: dict[str, torch.Tensor] | None = None,
) -> ImageTextChoiceTrainingResult:
    training_config = config or ImageTextChoiceTrainingConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)
    train_dataset = ImageTextChoiceDataset(train_examples)
    train_loader = _data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=True,
        device=device,
    )
    train_eval_loader = _data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=False,
        device=device,
    )
    eval_dataset: ImageTextChoiceDataset | None = None
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if eval_examples is not None:
        _validate_choice_set(train_examples, eval_examples)
        eval_dataset = ImageTextChoiceDataset(eval_examples)
        if eval_dataset.image_shape != train_dataset.image_shape:
            raise ValueError("eval images must have the same shape as train images")
        eval_loader = _data_loader(
            eval_dataset,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
            shuffle=False,
            device=device,
        )
    choices = _choices_from_examples(train_examples)
    prompt_options = (prompt, *additional_prompts)
    tokenizer_text = "\n".join((tokenizer_corpus, language_modeling_corpus or "", *prompt_options, *choices))
    tokenizer = tokenizer_override or build_text_tokenizer(
        tokenizer_text,
        kind="byte-pair",
        vocab_size=training_config.tokenizer_vocab_size,
    )
    prompt_token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    prompt_token_options = [torch.tensor(tokenizer.encode(value), dtype=torch.long) for value in prompt_options]
    choice_token_ids, choice_token_mask = _choice_token_tensors(choices, tokenizer)
    _validate_prompt_choice_lengths(prompt_token_options, choice_token_ids, training_config.text_context_length)
    text_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    text_eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if language_modeling_corpus is not None:
        text_token_ids = tokenizer.encode(language_modeling_corpus)
        text_dataset = LanguageModelingDataset(
            text_token_ids,
            context_length=training_config.text_context_length,
        )
        text_loader = _data_loader(
            text_dataset,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
            shuffle=True,
            device=device,
        )
        text_eval_loader = _data_loader(
            text_dataset,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
            shuffle=False,
            device=device,
        )
    preset = TRANSFORMER_CORE_PRESETS[training_config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=tokenizer.vocab_size,
        text_context_length=training_config.text_context_length,
        image_size=(train_dataset.image_shape[0], train_dataset.image_shape[1]),
        patch_size=training_config.image_patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        channel_count=train_dataset.channel_count,
    ).to(device)
    if initial_model_state_dict is not None:
        load_compatible_shared_state(model, initial_model_state_dict)
    prompt_token_ids = prompt_token_ids.to(device)
    prompt_token_options = [row.to(device) for row in prompt_token_options]
    choice_token_ids = choice_token_ids.to(device)
    choice_token_mask = choice_token_mask.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_adamw(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=training_config.lr_schedule,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )
    initial_loss = _loss(
        model,
        loss_fn,
        train_eval_loader,
        device,
        prompt_token_ids,
        choice_token_ids,
        choice_token_mask,
    )
    text_initial_loss = None
    if text_eval_loader is not None:
        text_initial_loss = _text_loss(model, loss_fn, text_eval_loader, device)
    image_iterator = iter(train_loader)
    text_iterator = iter(text_loader) if text_loader is not None else None
    model.train()
    for step in range(training_config.max_steps):
        optimizer.zero_grad(set_to_none=True)
        if text_iterator is not None and step % 2 == 0:
            try:
                batch_inputs, batch_targets = next(text_iterator)
            except StopIteration:
                text_iterator = iter(text_loader)
                batch_inputs, batch_targets = next(text_iterator)
            loss = _text_batch_loss(model, loss_fn, batch_inputs.to(device), batch_targets.to(device))
        else:
            try:
                batch_images, batch_labels = next(image_iterator)
            except StopIteration:
                image_iterator = iter(train_loader)
                batch_images, batch_labels = next(image_iterator)
            image_step = step if text_iterator is None else step // 2
            batch_prompt_token_ids = prompt_token_options[image_step % len(prompt_token_options)]
            loss = loss_fn(
                model.image_text_choice_logits(
                    batch_images.to(device),
                    batch_prompt_token_ids,
                    choice_token_ids,
                    choice_token_mask,
                ),
                batch_labels.to(device),
            )
        loss.backward()
        clip_gradients(model, training_config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    train_accuracy = _accuracy(
        model,
        train_eval_loader,
        device,
        prompt_token_ids,
        choice_token_ids,
        choice_token_mask,
    )
    eval_accuracy = None
    eval_count = 0
    if eval_dataset is not None and eval_loader is not None:
        eval_accuracy = _accuracy(
            model,
            eval_loader,
            device,
            prompt_token_ids,
            choice_token_ids,
            choice_token_mask,
        )
        eval_count = len(eval_dataset)
    text_final_loss = None
    if text_eval_loader is not None:
        text_final_loss = _text_loss(model, loss_fn, text_eval_loader, device)
    return ImageTextChoiceTrainingResult(
        metrics=ImageTextChoiceMetrics(
            train_case_count=len(train_dataset),
            eval_case_count=eval_count,
            train_initial_loss=initial_loss,
            train_final_loss=_loss(
                model,
                loss_fn,
                train_eval_loader,
                device,
                prompt_token_ids,
                choice_token_ids,
                choice_token_mask,
            ),
            train_accuracy=train_accuracy,
            eval_accuracy=eval_accuracy,
            text_initial_loss=text_initial_loss,
            text_final_loss=text_final_loss,
            max_steps=training_config.max_steps,
            model_preset=training_config.model_preset,
        ),
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        image_shape=train_dataset.image_shape,
    )


def evaluate_image_text_choice_model(
    *,
    model: SharedMultimodalModel,
    tokenizer: TextTokenizer,
    examples: list[ImageTextChoiceExample],
    prompt: str = "",
) -> ImageTextChoiceEvalMetrics:
    dataset = ImageTextChoiceDataset(examples)
    choices = _choices_from_examples(examples)
    prompt_token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    choice_token_ids, choice_token_mask = _choice_token_tensors(choices, tokenizer)
    device = next(model.parameters()).device
    data_loader = _data_loader(dataset, batch_size=len(dataset), seed=0, shuffle=False, device=device)
    prompt_token_ids = prompt_token_ids.to(device)
    choice_token_ids = choice_token_ids.to(device)
    choice_token_mask = choice_token_mask.to(device)
    loss_fn = nn.CrossEntropyLoss()
    return ImageTextChoiceEvalMetrics(
        case_count=len(dataset),
        loss=_loss(
            model,
            loss_fn,
            data_loader,
            device,
            prompt_token_ids,
            choice_token_ids,
            choice_token_mask,
        ),
        accuracy=_accuracy(
            model,
            data_loader,
            device,
            prompt_token_ids,
            choice_token_ids,
            choice_token_mask,
        ),
    )


def _choice_token_tensors(choices: tuple[str, ...], tokenizer: TextTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [tokenizer.encode(choice) for choice in choices]
    if any(not row for row in rows):
        raise ValueError("choice text must encode to at least one token")
    max_length = max(len(row) for row in rows)
    padded = [row + [0] * (max_length - len(row)) for row in rows]
    mask = [[True] * len(row) + [False] * (max_length - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _validate_prompt_choice_lengths(
    prompt_token_options: list[torch.Tensor],
    choice_token_ids: torch.Tensor,
    text_context_length: int,
) -> None:
    for prompt_token_ids in prompt_token_options:
        if prompt_token_ids.numel() + choice_token_ids.size(1) > text_context_length:
            raise ValueError("prompt plus choice token length must not exceed text_context_length")


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
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for batch_inputs, batch_targets in data_loader:
            loss = _text_batch_loss(model, loss_fn, batch_inputs.to(device), batch_targets.to(device))
            total_loss += float(loss.item())
            batch_count += 1
    if was_training:
        model.train()
    return total_loss / batch_count


def _loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    prompt_token_ids: torch.Tensor,
    choice_token_ids: torch.Tensor,
    choice_token_mask: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = loss_fn(
                model.image_text_choice_logits(
                    images,
                    prompt_token_ids,
                    choice_token_ids,
                    choice_token_mask,
                ),
                labels,
            )
            total_loss += float(loss.item()) * int(labels.numel())
            sample_count += int(labels.numel())
    if was_training:
        model.train()
    return total_loss / sample_count


def _accuracy(
    model: SharedMultimodalModel,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    prompt_token_ids: torch.Tensor,
    choice_token_ids: torch.Tensor,
    choice_token_mask: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model.image_text_choice_logits(
                images,
                prompt_token_ids,
                choice_token_ids,
                choice_token_mask,
            ).argmax(dim=1)
            correct_count += int((predictions == labels).sum().item())
            total_count += int(labels.numel())
    if was_training:
        model.train()
    return correct_count / total_count


def _validate_choice_set(
    train_examples: list[ImageTextChoiceExample],
    eval_examples: list[ImageTextChoiceExample],
) -> None:
    choices = train_examples[0].choices
    for example in eval_examples:
        if example.choices != choices:
            raise ValueError("eval examples must use the same choices as train examples")


def _choices_from_examples(examples: list[ImageTextChoiceExample]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("examples must not be empty")
    choices = examples[0].choices
    for example in examples:
        if example.choices != choices:
            raise ValueError("all examples must use the same choices")
    return choices


def _data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return seeded_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        device=device,
    )


def _validate_config(config: ImageTextChoiceTrainingConfig) -> None:
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
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if config.lr_schedule not in ("constant", "warmup_cosine"):
        raise ValueError("lr_schedule must be one of: constant, warmup_cosine")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.model_preset not in TRANSFORMER_CORE_PRESETS:
        raise ValueError(f"unknown model preset: {config.model_preset}")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
