from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from intrep.image_io import read_portable_image
from intrep.image_training_data import (
    channel_count_from_image_shape,
    image_tensor_from_path,
    seeded_data_loader,
)
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.shared_state_loading import load_compatible_shared_state
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer
from intrep.token_scoring import next_token_loss
from intrep.training_utils import LearningRateSchedule, build_adamw, build_lr_scheduler, clip_gradients


@dataclass(frozen=True)
class ImageTextAnswerExample:
    image_path: Path
    prompt: str
    answer_text: str

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if not self.answer_text:
            raise ValueError("answer_text must not be empty")


@dataclass(frozen=True)
class ImageTextAnswerTrainingConfig:
    text_context_length: int = 32
    image_patch_size: int = 4
    max_steps: int = 100
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
class ImageTextAnswerMetrics:
    train_case_count: int
    train_initial_loss: float
    train_final_loss: float
    max_steps: int
    model_preset: str


@dataclass(frozen=True)
class ImageTextAnswerTrainingResult:
    metrics: ImageTextAnswerMetrics
    model: SharedMultimodalModel
    tokenizer: TextTokenizer
    config: ImageTextAnswerTrainingConfig
    image_shape: tuple[int, ...]


class ImageTextAnswerDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples: list[ImageTextAnswerExample],
        text_token_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        if text_token_ids.size(0) != len(examples):
            raise ValueError("text_token_ids batch size must match examples")
        if loss_mask.shape != text_token_ids.shape:
            raise ValueError("loss_mask must have the same shape as text_token_ids")
        self.examples = tuple(examples)
        self.text_token_ids = text_token_ids
        self.loss_mask = loss_mask
        self.image_shape = tuple(int(value) for value in image_tensor_from_path(examples[0].image_path).shape)
        self.channel_count = channel_count_from_image_shape(self.image_shape)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = image_tensor_from_path(self.examples[index].image_path)
        if tuple(image.shape) != self.image_shape:
            raise ValueError("all images must have the same shape")
        return image, self.text_token_ids[index], self.loss_mask[index]


def train_image_text_answer_model(
    *,
    train_examples: list[ImageTextAnswerExample],
    tokenizer_corpus: str = "",
    config: ImageTextAnswerTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
    initial_model_state_dict: dict[str, torch.Tensor] | None = None,
) -> ImageTextAnswerTrainingResult:
    training_config = config or ImageTextAnswerTrainingConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)
    tokenizer_text = "\n".join(
        (
            tokenizer_corpus,
            *(example.prompt for example in train_examples),
            *(example.answer_text for example in train_examples),
        )
    )
    tokenizer = tokenizer_override or build_text_tokenizer(
        tokenizer_text,
        kind="byte-pair",
        vocab_size=training_config.tokenizer_vocab_size,
    )
    text_token_ids, loss_mask = _prompt_answer_token_tensors(train_examples, tokenizer)
    if text_token_ids.size(1) > training_config.text_context_length:
        raise ValueError("prompt plus answer token length must not exceed text_context_length")
    train_dataset = ImageTextAnswerDataset(train_examples, text_token_ids, loss_mask)
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
    initial_loss = _loss(model, train_eval_loader, device)
    train_iterator = iter(train_loader)
    model.train()
    for step in range(training_config.max_steps):
        try:
            batch_images, batch_token_ids, batch_loss_mask = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_images, batch_token_ids, batch_loss_mask = next(train_iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_tensor(
            model,
            batch_images.to(device),
            batch_token_ids.to(device),
            batch_loss_mask.to(device),
        )
        loss.backward()
        clip_gradients(model, training_config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    return ImageTextAnswerTrainingResult(
        metrics=ImageTextAnswerMetrics(
            train_case_count=len(train_dataset),
            train_initial_loss=initial_loss,
            train_final_loss=_loss(model, train_eval_loader, device),
            max_steps=training_config.max_steps,
            model_preset=training_config.model_preset,
        ),
        model=model,
        tokenizer=tokenizer,
        config=training_config,
        image_shape=train_dataset.image_shape,
    )


def image_text_answer_image_tensor_from_examples(examples: list[ImageTextAnswerExample]) -> torch.Tensor:
    images: list[np.ndarray] = []
    for example in examples:
        images.append(read_portable_image(example.image_path))
    if not images:
        raise ValueError("examples must not be empty")
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("all images must have the same shape")
    return torch.tensor(np.stack(images).astype(np.float32) / 255.0, dtype=torch.float32)


def load_image_text_answer_examples_jsonl(path: str | Path) -> list[ImageTextAnswerExample]:
    examples: list[ImageTextAnswerExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_text_answer_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-text-answer JSONL must contain at least one example")
    return examples


def image_text_answer_example_from_record(record: object, *, line_number: int) -> ImageTextAnswerExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: expected object")
    required = {"image_path", "prompt", "answer_text"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    prompt = record["prompt"]
    answer_text = record["answer_text"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: image_path must be a string")
    if not isinstance(prompt, str):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: prompt must be a string")
    if not isinstance(answer_text, str):
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: answer_text must be a string")
    try:
        return ImageTextAnswerExample(
            image_path=Path(image_path),
            prompt=prompt,
            answer_text=answer_text,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-text-answer JSONL at line {line_number}: {error}") from error


def image_text_answer_example_to_record(example: ImageTextAnswerExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "prompt": example.prompt,
        "answer_text": example.answer_text,
    }


def generate_image_text_answer(
    *,
    model: SharedMultimodalModel,
    tokenizer: TextTokenizer,
    image: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 8,
) -> str:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        image_batch = _image_batch(image).to(device)
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")
        for _ in range(max_new_tokens):
            text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            if text_ids.size(1) > model.text_context_length:
                break
            with torch.no_grad():
                logits = model.image_text_token_logits(image_batch, text_ids)
            next_token_id = int(logits[0, -1].argmax().item())
            token_ids.append(next_token_id)
        return tokenizer.decode(token_ids)[len(prompt):]
    finally:
        if was_training:
            model.train()


def _prompt_answer_token_tensors(
    examples: list[ImageTextAnswerExample],
    tokenizer: TextTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    rows = [tokenizer.encode(example.prompt + example.answer_text) for example in examples]
    prompt_lengths: list[int] = []
    for example, row in zip(examples, rows, strict=True):
        prompt_ids = tokenizer.encode(example.prompt)
        if not prompt_ids:
            raise ValueError("prompt must encode to at least one token")
        if len(row) <= len(prompt_ids):
            raise ValueError("answer_text must encode to at least one token")
        prompt_lengths.append(len(prompt_ids))
    max_length = max(len(row) for row in rows)
    padded = [row + [0] * (max_length - len(row)) for row in rows]
    masks = [
        [False] * prompt_length + [True] * (len(row) - prompt_length) + [False] * (max_length - len(row))
        for row, prompt_length in zip(rows, prompt_lengths, strict=True)
    ]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def _loss(
    model: SharedMultimodalModel,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, text_token_ids, loss_mask in data_loader:
            loss = _loss_tensor(
                model,
                images.to(device),
                text_token_ids.to(device),
                loss_mask.to(device),
            )
            batch_size = int(text_token_ids.size(0))
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
    if was_training:
        model.train()
    return total_loss / sample_count


def _loss_tensor(
    model: SharedMultimodalModel,
    images: torch.Tensor,
    text_token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    logits = model.image_text_token_logits(images, text_token_ids)
    image_token_count = logits.size(1) - text_token_ids.size(1)
    targets = torch.zeros((text_token_ids.size(0), logits.size(1)), dtype=torch.long, device=text_token_ids.device)
    targets[:, image_token_count:] = text_token_ids
    full_loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    full_loss_mask[:, image_token_count:] = loss_mask
    return next_token_loss(logits, targets, loss_mask=full_loss_mask)


def _image_batch(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 2:
        return image.unsqueeze(0)
    if image.ndim == 3 and image.size(-1) in (1, 3):
        return image.unsqueeze(0)
    if image.ndim == 3 and image.size(0) == 1:
        return image
    raise ValueError("image must have shape [height, width], [height, width, channels], or [1, height, width]")


def _data_loader(
    dataset: ImageTextAnswerDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return seeded_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        device=device,
    )


def _validate_config(config: ImageTextAnswerTrainingConfig) -> None:
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
