from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from intrep.image_io import read_portable_image
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer
from intrep.token_scoring import next_token_loss


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


def train_image_text_answer_model(
    *,
    train_examples: list[ImageTextAnswerExample],
    tokenizer_corpus: str = "",
    config: ImageTextAnswerTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> ImageTextAnswerTrainingResult:
    training_config = config or ImageTextAnswerTrainingConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)
    train_images = image_text_answer_image_tensor_from_examples(train_examples)
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
        channel_count=_channel_count_from_images(train_images),
    ).to(device)
    train_images = train_images.to(device)
    text_token_ids = text_token_ids.to(device)
    loss_mask = loss_mask.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    initial_loss = _loss(model, train_images, text_token_ids, loss_mask)
    for step in range(training_config.max_steps):
        start = (step * training_config.batch_size) % len(train_images)
        indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
        batch_images = train_images.index_select(0, indices)
        batch_token_ids = text_token_ids.index_select(0, indices)
        batch_loss_mask = loss_mask.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_tensor(model, batch_images, batch_token_ids, batch_loss_mask)
        loss.backward()
        optimizer.step()

    return ImageTextAnswerTrainingResult(
        metrics=ImageTextAnswerMetrics(
            train_case_count=int(train_images.size(0)),
            train_initial_loss=initial_loss,
            train_final_loss=_loss(model, train_images, text_token_ids, loss_mask),
            max_steps=training_config.max_steps,
            model_preset=training_config.model_preset,
        ),
        model=model,
        tokenizer=tokenizer,
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
    images: torch.Tensor,
    text_token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = _loss_tensor(model, images, text_token_ids, loss_mask)
    if was_training:
        model.train()
    return float(loss.item())


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


def _channel_count_from_images(images: torch.Tensor) -> int:
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[3])
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")


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
    if config.model_preset not in TRANSFORMER_CORE_PRESETS:
        raise ValueError(f"unknown model preset: {config.model_preset}")
    if config.device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer_vocab_size must be positive")
