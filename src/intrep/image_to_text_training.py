from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import CausalTextModel, build_causal_text_config
from intrep.image_classification import ImageChoiceExample, ImagePatchInputLayer, image_label_tensors_from_examples
from intrep.language_modeling_training import resolve_training_device
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.text_tokenizer import TextTokenizer
from intrep.token_scoring import next_token_loss


@dataclass(frozen=True)
class ImageToTextTrainingConfig:
    patch_size: int = 4
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "tiny"
    device: str = "auto"


@dataclass(frozen=True)
class ImageToTextMetrics:
    target: str
    input_representation: str
    output_representation: str
    train_case_count: int
    eval_case_count: int
    train_initial_loss: float
    train_final_loss: float
    eval_initial_loss: float | None
    eval_final_loss: float | None
    patch_size: int
    max_steps: int
    model_preset: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "input_representation": self.input_representation,
            "output_representation": self.output_representation,
            "train_case_count": self.train_case_count,
            "eval_case_count": self.eval_case_count,
            "train_initial_loss": self.train_initial_loss,
            "train_final_loss": self.train_final_loss,
            "eval_initial_loss": self.eval_initial_loss,
            "eval_final_loss": self.eval_final_loss,
            "image_patch_size": self.patch_size,
            "max_steps": self.max_steps,
            "model_preset": self.model_preset,
        }


def train_image_to_text_labels(
    *,
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample] | None = None,
    config: ImageToTextTrainingConfig | None = None,
    tokenizer: TextTokenizer | None = None,
) -> ImageToTextMetrics:
    training_config = config or ImageToTextTrainingConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)  # type: ignore[arg-type]
    text_tokenizer = tokenizer or ByteTokenizer()
    train_images, train_labels = image_label_tensors_from_examples(train_examples)
    train_token_ids, train_token_mask = _label_token_tensors(train_examples, text_tokenizer)
    eval_images: torch.Tensor | None = None
    eval_token_ids: torch.Tensor | None = None
    eval_token_mask: torch.Tensor | None = None
    if eval_examples is not None:
        eval_images, _ = image_label_tensors_from_examples(eval_examples)
        if tuple(eval_images.shape[1:]) != tuple(train_images.shape[1:]):
            raise ValueError("eval images must have the same shape as train images")
        eval_token_ids, eval_token_mask = _label_token_tensors(eval_examples, text_tokenizer)

    preset = TRANSFORMER_CORE_PRESETS[training_config.model_preset]
    image_input_layer = ImagePatchInputLayer(
        image_size=(int(train_images.shape[1]), int(train_images.shape[2])),
        patch_size=training_config.patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        channel_count=_channel_count_from_images(train_images),
    ).to(device)
    image_sequence_length = image_input_layer.position_embedding.num_embeddings
    text_context_length = image_sequence_length + train_token_ids.size(1)
    text_model = CausalTextModel(
        build_causal_text_config(
            preset=training_config.model_preset,
            vocab_size=text_tokenizer.vocab_size,
            context_length=text_context_length,
        )
    ).to(device)
    train_images = train_images.to(device)
    train_token_ids = train_token_ids.to(device)
    train_token_mask = train_token_mask.to(device)
    if eval_images is not None and eval_token_ids is not None:
        eval_images = eval_images.to(device)
        eval_token_ids = eval_token_ids.to(device)
        assert eval_token_mask is not None
        eval_token_mask = eval_token_mask.to(device)

    optimizer = torch.optim.AdamW(
        list(image_input_layer.parameters()) + list(text_model.parameters()),
        lr=training_config.learning_rate,
    )
    train_initial_loss = _image_to_text_loss(
        image_input_layer=image_input_layer,
        text_model=text_model,
        images=train_images,
        token_ids=train_token_ids,
        token_mask=train_token_mask,
    )
    eval_initial_loss = None
    if eval_images is not None and eval_token_ids is not None and eval_token_mask is not None:
        eval_initial_loss = _image_to_text_loss(
            image_input_layer=image_input_layer,
            text_model=text_model,
            images=eval_images,
            token_ids=eval_token_ids,
            token_mask=eval_token_mask,
        )

    image_input_layer.train()
    text_model.train()
    for step in range(training_config.max_steps):
        start = (step * training_config.batch_size) % len(train_images)
        indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
        batch_images = train_images.index_select(0, indices)
        batch_token_ids = train_token_ids.index_select(0, indices)
        batch_token_mask = train_token_mask.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = _image_to_text_loss_tensor(
            image_input_layer=image_input_layer,
            text_model=text_model,
            images=batch_images,
            token_ids=batch_token_ids,
            token_mask=batch_token_mask,
        )
        loss.backward()
        optimizer.step()

    train_final_loss = _image_to_text_loss(
        image_input_layer=image_input_layer,
        text_model=text_model,
        images=train_images,
        token_ids=train_token_ids,
        token_mask=train_token_mask,
    )
    eval_final_loss = None
    eval_count = 0
    if eval_images is not None and eval_token_ids is not None and eval_token_mask is not None:
        eval_final_loss = _image_to_text_loss(
            image_input_layer=image_input_layer,
            text_model=text_model,
            images=eval_images,
            token_ids=eval_token_ids,
            token_mask=eval_token_mask,
        )
        eval_count = len(eval_examples or [])
    return ImageToTextMetrics(
        target="answer_text",
        input_representation="image-patches",
        output_representation="text-tokens",
        train_case_count=len(train_examples),
        eval_case_count=eval_count,
        train_initial_loss=train_initial_loss,
        train_final_loss=train_final_loss,
        eval_initial_loss=eval_initial_loss,
        eval_final_loss=eval_final_loss,
        patch_size=training_config.patch_size,
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )


def _image_to_text_loss(
    *,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> float:
    image_input_layer.eval()
    text_model.eval()
    with torch.no_grad():
        return float(
            _image_to_text_loss_tensor(
                image_input_layer=image_input_layer,
                text_model=text_model,
                images=images,
                token_ids=token_ids,
                token_mask=token_mask,
            ).item()
        )


def _image_to_text_loss_tensor(
    *,
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    images: torch.Tensor,
    token_ids: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    image_embeddings = image_input_layer(images)
    text_embeddings = text_model.embed_tokens(token_ids, position_offset=image_embeddings.size(1))
    combined_embeddings = concatenate_input_embedding_sequences(image_embeddings, text_embeddings)
    hidden = text_model.encode_embeddings(combined_embeddings, causal=True)
    logits = text_model.token_logits(hidden)
    targets = torch.zeros(
        (token_ids.size(0), combined_embeddings.size(1)),
        dtype=torch.long,
        device=token_ids.device,
    )
    text_start = image_embeddings.size(1)
    targets[:, text_start:] = token_ids
    loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_mask[:, text_start:] = token_mask
    return next_token_loss(logits, targets, loss_mask=loss_mask)


def _label_token_tensors(examples: list[ImageChoiceExample], tokenizer: TextTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("examples must not be empty")
    rows = [tokenizer.encode(example.answer_text) for example in examples]
    if any(not row for row in rows):
        raise ValueError("answer_text must encode to at least one token")
    max_length = max(len(row) for row in rows)
    padded = [row + [0] * (max_length - len(row)) for row in rows]
    mask = [[True] * len(row) + [False] * (max_length - len(row)) for row in rows]
    return torch.tensor(padded, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _channel_count_from_images(images: torch.Tensor) -> int:
    if images.ndim == 3:
        return 1
    if images.ndim == 4:
        return int(images.shape[3])
    raise ValueError("images must have shape [batch, height, width] or [batch, height, width, channels]")


def _validate_config(config: ImageToTextTrainingConfig) -> None:
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


def write_metrics(path: str | Path, metrics: ImageToTextMetrics) -> None:
    Path(path).write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
