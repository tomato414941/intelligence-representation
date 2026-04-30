from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from intrep.byte_tokenizer import ByteTokenizer
from intrep.causal_text_model import (
    CausalTextConfig,
    CausalTextModel,
    build_causal_text_config,
    causal_text_config_to_dict,
)
from intrep.image_classification import ImageChoiceExample, ImagePatchInputLayer, image_label_tensors_from_examples
from intrep.image_conditioned_text_evaluation import (
    ImageConditionedTextChoiceMetrics,
    evaluate_image_conditioned_text_choices,
)
from intrep.language_modeling_training import resolve_training_device
from intrep.model_input import concatenate_input_embedding_sequences
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.text_tokenizer import TextTokenizer, text_tokenizer_from_payload, text_tokenizer_to_payload
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
    choice_eval_limit: int | None = 200


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
    train_choice_case_count: int
    eval_choice_case_count: int
    train_choice_accuracy: float | None
    eval_choice_accuracy: float | None
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
            "train_choice_case_count": self.train_choice_case_count,
            "eval_choice_case_count": self.eval_choice_case_count,
            "train_choice_accuracy": self.train_choice_accuracy,
            "eval_choice_accuracy": self.eval_choice_accuracy,
            "image_patch_size": self.patch_size,
            "max_steps": self.max_steps,
            "model_preset": self.model_preset,
        }


@dataclass(frozen=True)
class ImageToTextTrainingResult:
    metrics: ImageToTextMetrics
    image_input_layer: ImagePatchInputLayer
    text_model: CausalTextModel
    tokenizer: TextTokenizer
    image_shape: tuple[int, ...]
    config: ImageToTextTrainingConfig


@dataclass(frozen=True)
class ImageToTextCheckpoint:
    image_input_layer: ImagePatchInputLayer
    text_model: CausalTextModel
    tokenizer: TextTokenizer
    image_shape: tuple[int, ...]
    metrics: dict[str, object]


def train_image_to_text_labels(
    *,
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample] | None = None,
    config: ImageToTextTrainingConfig | None = None,
    tokenizer: TextTokenizer | None = None,
) -> ImageToTextMetrics:
    return train_image_to_text_labels_with_result(
        train_examples=train_examples,
        eval_examples=eval_examples,
        config=config,
        tokenizer=tokenizer,
    ).metrics


def train_image_to_text_labels_with_result(
    *,
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample] | None = None,
    config: ImageToTextTrainingConfig | None = None,
    tokenizer: TextTokenizer | None = None,
) -> ImageToTextTrainingResult:
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
    train_choice_examples = _choice_eval_examples(train_examples, training_config.choice_eval_limit)
    eval_choice_examples = _choice_eval_examples(eval_examples or [], training_config.choice_eval_limit)
    train_choice_accuracy = evaluate_image_conditioned_text_choices(
        examples=train_choice_examples,
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=text_tokenizer,
        prompt="",
    ).accuracy
    eval_choice_accuracy = None
    if eval_choice_examples:
        eval_choice_accuracy = evaluate_image_conditioned_text_choices(
            examples=eval_choice_examples,
            image_input_layer=image_input_layer,
            text_model=text_model,
            tokenizer=text_tokenizer,
            prompt="",
        ).accuracy
    metrics = ImageToTextMetrics(
        target="answer_text",
        input_representation="image-patches",
        output_representation="text-tokens",
        train_case_count=len(train_examples),
        eval_case_count=eval_count,
        train_initial_loss=train_initial_loss,
        train_final_loss=train_final_loss,
        eval_initial_loss=eval_initial_loss,
        eval_final_loss=eval_final_loss,
        train_choice_case_count=len(train_choice_examples),
        eval_choice_case_count=len(eval_choice_examples),
        train_choice_accuracy=train_choice_accuracy,
        eval_choice_accuracy=eval_choice_accuracy,
        patch_size=training_config.patch_size,
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )
    return ImageToTextTrainingResult(
        metrics=metrics,
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=text_tokenizer,
        image_shape=tuple(int(value) for value in train_images.shape[1:]),
        config=training_config,
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
    if config.choice_eval_limit is not None and config.choice_eval_limit <= 0:
        raise ValueError("choice_eval_limit must be positive")


def _choice_eval_examples(examples: list[ImageChoiceExample], limit: int | None) -> list[ImageChoiceExample]:
    if limit is None:
        return examples
    return examples[:limit]


def write_metrics(path: str | Path, metrics: ImageToTextMetrics) -> None:
    Path(path).write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_image_to_text_checkpoint(path: str | Path, result: ImageToTextTrainingResult) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "intrep.model_checkpoint.v1",
            "task": "image-to-text",
            "image_input_layer": result.image_input_layer.state_dict(),
            "text_model": result.text_model.state_dict(),
            "tokenizer": text_tokenizer_to_payload(result.tokenizer),
            "config": asdict(result.config),
            "model_config": causal_text_config_to_dict(result.text_model.config),
            "metrics": result.metrics.to_dict(),
            "image_shape": result.image_shape,
        },
        checkpoint_path,
    )


def load_image_to_text_checkpoint(path: str | Path, *, device: str = "auto") -> ImageToTextCheckpoint:
    resolved_device = resolve_training_device(device)  # type: ignore[arg-type]
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    if payload.get("schema_version") != "intrep.model_checkpoint.v1":
        raise ValueError("unsupported checkpoint schema")
    if payload.get("task") != "image-to-text":
        raise ValueError("checkpoint task must be image-to-text")

    tokenizer_payload = payload.get("tokenizer")
    if not isinstance(tokenizer_payload, dict):
        raise ValueError("checkpoint requires tokenizer payload")
    tokenizer = text_tokenizer_from_payload(tokenizer_payload)

    model_config_payload = payload.get("model_config")
    if not isinstance(model_config_payload, dict):
        raise ValueError("checkpoint requires model_config")
    text_config = CausalTextConfig(**model_config_payload)
    if text_config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint tokenizer vocab size does not match model vocab size")
    text_model = CausalTextModel(text_config).to(resolved_device)
    text_model.load_state_dict(payload["text_model"])

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint requires training config")
    image_shape = _image_shape_from_payload(payload.get("image_shape"))
    image_input_layer = ImagePatchInputLayer(
        image_size=(image_shape[0], image_shape[1]),
        patch_size=int(config_payload["patch_size"]),
        embedding_dim=text_config.embedding_dim,
        channel_count=1 if len(image_shape) == 2 else image_shape[2],
    ).to(resolved_device)
    image_input_layer.load_state_dict(payload["image_input_layer"])

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return ImageToTextCheckpoint(
        image_input_layer=image_input_layer,
        text_model=text_model,
        tokenizer=tokenizer,
        image_shape=image_shape,
        metrics=metrics,
    )


def evaluate_image_to_text_checkpoint(
    *,
    checkpoint_path: str | Path,
    examples: list[ImageChoiceExample],
    prompt: str = "",
    device: str = "auto",
) -> ImageConditionedTextChoiceMetrics:
    checkpoint = load_image_to_text_checkpoint(checkpoint_path, device=device)
    return evaluate_image_conditioned_text_choices(
        examples=examples,
        image_input_layer=checkpoint.image_input_layer,
        text_model=checkpoint.text_model,
        tokenizer=checkpoint.tokenizer,
        prompt=prompt,
    )


def _image_shape_from_payload(payload: object) -> tuple[int, ...]:
    if (
        isinstance(payload, (list, tuple))
        and len(payload) in (2, 3)
        and all(isinstance(value, int) for value in payload)
    ):
        return tuple(payload)
    raise ValueError("checkpoint requires image_shape")
