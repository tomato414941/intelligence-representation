from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.image_classification import ImageChoiceExample, image_label_tensors_from_examples
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.text_tokenizer import TextTokenizer, build_text_tokenizer


@dataclass(frozen=True)
class ImageTextCandidateTrainingConfig:
    text_context_length: int = 16
    image_patch_size: int = 4
    max_steps: int = 1000
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "tiny"
    device: LanguageModelingTrainingDevice = "cpu"
    tokenizer_vocab_size: int = 512


@dataclass(frozen=True)
class ImageTextCandidateMetrics:
    train_case_count: int
    eval_case_count: int
    train_initial_loss: float
    train_final_loss: float
    train_accuracy: float
    eval_accuracy: float | None
    max_steps: int
    model_preset: str


@dataclass(frozen=True)
class ImageTextCandidateTrainingResult:
    metrics: ImageTextCandidateMetrics
    model: SharedMultimodalModel
    tokenizer: TextTokenizer


@dataclass(frozen=True)
class ImageTextCandidateEvalMetrics:
    case_count: int
    loss: float
    accuracy: float


def train_image_text_candidate_model(
    *,
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample] | None = None,
    text_corpus: str = "",
    prompt: str = "",
    config: ImageTextCandidateTrainingConfig | None = None,
    tokenizer_override: TextTokenizer | None = None,
) -> ImageTextCandidateTrainingResult:
    training_config = config or ImageTextCandidateTrainingConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)
    train_images, train_labels = image_label_tensors_from_examples(train_examples)
    if eval_examples is not None:
        _validate_choice_set(train_examples, eval_examples)
        eval_images, eval_labels = image_label_tensors_from_examples(eval_examples)
    else:
        eval_images, eval_labels = None, None
    choices = _choices_from_examples(train_examples)
    tokenizer_text = "\n".join((text_corpus, prompt, *choices))
    tokenizer = tokenizer_override or build_text_tokenizer(
        tokenizer_text,
        kind="byte-pair",
        vocab_size=training_config.tokenizer_vocab_size,
    )
    prompt_token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    candidate_token_ids, candidate_token_mask = _candidate_token_tensors(choices, tokenizer)
    if prompt_token_ids.numel() + candidate_token_ids.size(1) > training_config.text_context_length:
        raise ValueError("prompt plus candidate token length must not exceed text_context_length")
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
    train_labels = train_labels.to(device)
    prompt_token_ids = prompt_token_ids.to(device)
    candidate_token_ids = candidate_token_ids.to(device)
    candidate_token_mask = candidate_token_mask.to(device)
    if eval_images is not None and eval_labels is not None:
        eval_images = eval_images.to(device)
        eval_labels = eval_labels.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    initial_loss = _loss(
        model,
        loss_fn,
        train_images,
        train_labels,
        prompt_token_ids,
        candidate_token_ids,
        candidate_token_mask,
    )
    for step in range(training_config.max_steps):
        start = (step * training_config.batch_size) % len(train_images)
        indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
        batch_images = train_images.index_select(0, indices)
        batch_labels = train_labels.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(
            model.image_text_fusion_candidate_logits(
                batch_images,
                prompt_token_ids,
                candidate_token_ids,
                candidate_token_mask,
            ),
            batch_labels,
        )
        loss.backward()
        optimizer.step()

    train_accuracy = _accuracy(
        model,
        train_images,
        train_labels,
        prompt_token_ids,
        candidate_token_ids,
        candidate_token_mask,
    )
    eval_accuracy = None
    eval_count = 0
    if eval_images is not None and eval_labels is not None:
        eval_accuracy = _accuracy(
            model,
            eval_images,
            eval_labels,
            prompt_token_ids,
            candidate_token_ids,
            candidate_token_mask,
        )
        eval_count = int(eval_labels.numel())
    return ImageTextCandidateTrainingResult(
        metrics=ImageTextCandidateMetrics(
            train_case_count=int(train_labels.numel()),
            eval_case_count=eval_count,
            train_initial_loss=initial_loss,
            train_final_loss=_loss(
                model,
                loss_fn,
                train_images,
                train_labels,
                prompt_token_ids,
                candidate_token_ids,
                candidate_token_mask,
            ),
            train_accuracy=train_accuracy,
            eval_accuracy=eval_accuracy,
            max_steps=training_config.max_steps,
            model_preset=training_config.model_preset,
        ),
        model=model,
        tokenizer=tokenizer,
    )


def evaluate_image_text_candidate_model(
    *,
    model: SharedMultimodalModel,
    tokenizer: TextTokenizer,
    examples: list[ImageChoiceExample],
    prompt: str = "",
) -> ImageTextCandidateEvalMetrics:
    images, labels = image_label_tensors_from_examples(examples)
    choices = _choices_from_examples(examples)
    prompt_token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
    candidate_token_ids, candidate_token_mask = _candidate_token_tensors(choices, tokenizer)
    device = next(model.parameters()).device
    images = images.to(device)
    labels = labels.to(device)
    prompt_token_ids = prompt_token_ids.to(device)
    candidate_token_ids = candidate_token_ids.to(device)
    candidate_token_mask = candidate_token_mask.to(device)
    loss_fn = nn.CrossEntropyLoss()
    return ImageTextCandidateEvalMetrics(
        case_count=int(labels.numel()),
        loss=_loss(
            model,
            loss_fn,
            images,
            labels,
            prompt_token_ids,
            candidate_token_ids,
            candidate_token_mask,
        ),
        accuracy=_accuracy(
            model,
            images,
            labels,
            prompt_token_ids,
            candidate_token_ids,
            candidate_token_mask,
        ),
    )


def _candidate_token_tensors(choices: tuple[str, ...], tokenizer: TextTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [tokenizer.encode(choice) for choice in choices]
    if any(not row for row in rows):
        raise ValueError("candidate text must encode to at least one token")
    max_length = max(len(row) for row in rows)
    padded = [row + [0] * (max_length - len(row)) for row in rows]
    mask = [[True] * len(row) + [False] * (max_length - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_token_mask: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = loss_fn(
            model.image_text_fusion_candidate_logits(
                images,
                prompt_token_ids,
                candidate_token_ids,
                candidate_token_mask,
            ),
            labels,
        )
    if was_training:
        model.train()
    return float(loss.item())


def _accuracy(
    model: SharedMultimodalModel,
    images: torch.Tensor,
    labels: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_token_mask: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions = model.image_text_fusion_candidate_logits(
            images,
            prompt_token_ids,
            candidate_token_ids,
            candidate_token_mask,
        ).argmax(dim=1)
    if was_training:
        model.train()
    return float((predictions == labels).float().mean().item())


def _validate_choice_set(
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample],
) -> None:
    choices = train_examples[0].choices
    for example in eval_examples:
        if example.choices != choices:
            raise ValueError("eval examples must use the same choices as train examples")


def _choices_from_examples(examples: list[ImageChoiceExample]) -> tuple[str, ...]:
    if not examples:
        raise ValueError("examples must not be empty")
    choices = examples[0].choices
    for example in examples:
        if example.choices != choices:
            raise ValueError("all examples must use the same choices")
    return choices


def _channel_count_from_images(images: torch.Tensor) -> int:
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[3])
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")


def _validate_config(config: ImageTextCandidateTrainingConfig) -> None:
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
