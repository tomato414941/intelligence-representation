from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from intrep.vision.classification_checkpoint import load_image_classification_checkpoint
from intrep.problems.image_text_answer.checkpoint import load_image_text_answer_checkpoint
from intrep.problems.image_text_choice.checkpoint import load_image_text_choice_checkpoint
from intrep.text.language_modeling_training import LanguageModelingTrainingDevice, resolve_training_device
from intrep.text.tokenizer import TextTokenizer


@dataclass(frozen=True)
class SharedCoreInitialization:
    model_state_dict: dict[str, torch.Tensor]
    tokenizer: TextTokenizer | None
    source_schema: str


def load_shared_core_initialization(
    path: str | Path,
    *,
    device: LanguageModelingTrainingDevice = "auto",
) -> SharedCoreInitialization:
    """Load compatible state from any checkpoint using ImageTextSharedModel."""
    resolved_device = resolve_training_device(device)
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    schema = payload.get("schema_version")
    if schema == "intrep.image_text_choice_checkpoint.v1":
        checkpoint = load_image_text_choice_checkpoint(checkpoint_path, device=device)
        tokenizer: TextTokenizer | None = checkpoint.tokenizer
    elif schema == "intrep.image_text_answer_checkpoint.v1":
        checkpoint = load_image_text_answer_checkpoint(checkpoint_path, device=device)
        tokenizer = checkpoint.tokenizer
    elif schema == "intrep.image_classification_checkpoint.v1":
        checkpoint = load_image_classification_checkpoint(checkpoint_path, device=device)
        tokenizer = None
    else:
        raise ValueError("checkpoint is not a shared core checkpoint")
    return SharedCoreInitialization(
        model_state_dict=checkpoint.model.state_dict(),
        tokenizer=tokenizer,
        source_schema=schema,
    )
