from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intrep.image_text_answer_training import ImageTextAnswerTrainingConfig, ImageTextAnswerTrainingResult
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel
from intrep.text_tokenizer import TextTokenizer, text_tokenizer_from_payload, text_tokenizer_to_payload


@dataclass(frozen=True)
class ImageTextAnswerCheckpoint:
    model: SharedMultimodalModel
    tokenizer: TextTokenizer
    config: ImageTextAnswerTrainingConfig
    image_shape: tuple[int, ...]
    metrics: dict[str, object]


def save_image_text_answer_checkpoint(path: str | Path, result: ImageTextAnswerTrainingResult) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "intrep.image_text_answer_checkpoint.v1",
            "model": result.model.state_dict(),
            "tokenizer": text_tokenizer_to_payload(result.tokenizer),
            "config": asdict(result.config),
            "image_shape": result.image_shape,
            "metrics": asdict(result.metrics),
        },
        checkpoint_path,
    )


def load_image_text_answer_checkpoint(
    path: str | Path,
    *,
    device: LanguageModelingTrainingDevice = "auto",
) -> ImageTextAnswerCheckpoint:
    resolved_device = resolve_training_device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    if payload.get("schema_version") != "intrep.image_text_answer_checkpoint.v1":
        raise ValueError("unsupported image/text answer checkpoint schema")
    tokenizer_payload = payload.get("tokenizer")
    if not isinstance(tokenizer_payload, dict):
        raise ValueError("checkpoint requires tokenizer payload")
    tokenizer = text_tokenizer_from_payload(tokenizer_payload)
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint requires config")
    config = ImageTextAnswerTrainingConfig(**config_payload)
    image_shape = _image_shape_from_payload(payload.get("image_shape"))
    preset = TRANSFORMER_CORE_PRESETS[config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=tokenizer.vocab_size,
        text_context_length=config.text_context_length,
        image_size=(image_shape[0], image_shape[1]),
        patch_size=config.image_patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        channel_count=1 if len(image_shape) == 2 else image_shape[2],
    ).to(resolved_device)
    model.load_state_dict(payload["model"])
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return ImageTextAnswerCheckpoint(
        model=model,
        tokenizer=tokenizer,
        config=config,
        image_shape=image_shape,
        metrics=metrics,
    )


def _image_shape_from_payload(payload: object) -> tuple[int, ...]:
    if (
        isinstance(payload, (list, tuple))
        and len(payload) in (2, 3)
        and all(isinstance(value, int) for value in payload)
    ):
        return tuple(payload)
    raise ValueError("checkpoint requires image_shape")
