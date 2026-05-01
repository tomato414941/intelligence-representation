from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from intrep.image_classification import (
    ImageClassificationConfig,
    ImageClassificationTrainingResult,
)
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import SharedMultimodalModel


@dataclass(frozen=True)
class ImageClassificationCheckpoint:
    model: SharedMultimodalModel
    config: ImageClassificationConfig
    image_shape: tuple[int, ...]
    label_names: tuple[str, ...]
    metrics: dict[str, object]


def save_image_classification_checkpoint(path: str | Path, result: ImageClassificationTrainingResult) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "intrep.image_classification_checkpoint.v1",
            "model": result.model.state_dict(),
            "config": asdict(result.config),
            "image_shape": result.image_shape,
            "label_names": result.label_names,
            "metrics": result.metrics.to_dict(),
        },
        checkpoint_path,
    )


def load_image_classification_checkpoint(
    path: str | Path,
    *,
    device: LanguageModelingTrainingDevice = "auto",
) -> ImageClassificationCheckpoint:
    resolved_device = resolve_training_device(device)
    payload = torch.load(Path(path), map_location=resolved_device, weights_only=False)
    if payload.get("schema_version") != "intrep.image_classification_checkpoint.v1":
        raise ValueError("unsupported image classification checkpoint schema")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise ValueError("checkpoint requires config")
    config = ImageClassificationConfig(**config_payload)
    image_shape = _image_shape_from_payload(payload.get("image_shape"))
    label_names = _label_names_from_payload(payload.get("label_names"))
    preset = TRANSFORMER_CORE_PRESETS[config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=1,
        text_context_length=1,
        image_size=(image_shape[0], image_shape[1]),
        patch_size=config.patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        channel_count=1 if len(image_shape) == 2 else image_shape[2],
        num_classes=len(label_names),
    ).to(resolved_device)
    model.load_state_dict(payload["model"])
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return ImageClassificationCheckpoint(
        model=model,
        config=config,
        image_shape=image_shape,
        label_names=label_names,
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


def _label_names_from_payload(payload: object) -> tuple[str, ...]:
    if isinstance(payload, (list, tuple)) and payload and all(isinstance(value, str) for value in payload):
        return tuple(payload)
    raise ValueError("checkpoint requires label_names")
