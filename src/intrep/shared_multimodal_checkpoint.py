from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from intrep.image_text_answer_checkpoint import load_image_text_answer_checkpoint
from intrep.image_text_choice_checkpoint import load_image_text_choice_checkpoint
from intrep.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.text_tokenizer import TextTokenizer


@dataclass(frozen=True)
class SharedMultimodalInitialization:
    model_state_dict: dict[str, torch.Tensor]
    tokenizer: TextTokenizer
    source_schema: str


def load_shared_multimodal_initialization(
    path: str | Path,
    *,
    device: LanguageModelingTrainingDevice = "auto",
) -> SharedMultimodalInitialization:
    """Load weights/tokenizer from any checkpoint using SharedMultimodalModel."""
    resolved_device = resolve_training_device(device)
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    schema = payload.get("schema_version")
    if schema == "intrep.image_text_choice_checkpoint.v1":
        checkpoint = load_image_text_choice_checkpoint(checkpoint_path, device=device)
    elif schema == "intrep.image_text_answer_checkpoint.v1":
        checkpoint = load_image_text_answer_checkpoint(checkpoint_path, device=device)
    else:
        raise ValueError("checkpoint is not a shared multimodal checkpoint")
    return SharedMultimodalInitialization(
        model_state_dict=checkpoint.model.state_dict(),
        tokenizer=checkpoint.tokenizer,
        source_schema=schema,
    )
