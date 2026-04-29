from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intrep.causal_text_model import CausalTextConfig
from intrep.fashion_mnist_vit import (
    ImageChoiceExample,
    ImageClassificationConfig,
    ImageClassificationTrainingResult,
    train_fashion_mnist_classifier_with_result,
)
from intrep.image_text_training import (
    ImageTextTrainingConfig,
    ImageTextTrainingResult,
    image_text_examples_from_choices,
    train_image_text_examples,
)
from intrep.language_modeling_training import (
    LanguageModelingTrainingArtifacts,
    LanguageModelingTrainingConfig,
    train_language_modeling_with_artifacts,
)
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.text_examples import LanguageModelingExample


@dataclass(frozen=True)
class ImageTextTrainingPhasesResult:
    image_classification: ImageClassificationTrainingResult
    language_modeling: LanguageModelingTrainingArtifacts
    image_text: ImageTextTrainingResult


def run_image_text_training_phases(
    *,
    image_examples: Sequence[ImageChoiceExample],
    text_examples: Sequence[LanguageModelingExample],
    image_classification_config: ImageClassificationConfig | None = None,
    language_modeling_config: LanguageModelingTrainingConfig | None = None,
    image_text_config: ImageTextTrainingConfig | None = None,
    prompt: str = "label: ",
) -> ImageTextTrainingPhasesResult:
    if not image_examples:
        raise ValueError("image_examples must not be empty")
    if not text_examples:
        raise ValueError("text_examples must not be empty")

    image_config = image_classification_config or ImageClassificationConfig()
    text_config = language_modeling_config or LanguageModelingTrainingConfig()
    image_result = train_fashion_mnist_classifier_with_result(
        train_examples=list(image_examples),
        config=image_config,
    )
    text_artifacts = train_language_modeling_with_artifacts(
        train_examples=tuple(text_examples),
        training_config=text_config,
        model_config=CausalTextConfig(
            vocab_size=_model_vocab_size(text_config),
            context_length=text_config.context_length,
            **_core_shape(),
        ),
    )
    _validate_embedding_dim(
        image_result.model.image_input_layer.patch_embedding.out_features,
        text_artifacts.model.config.embedding_dim,
    )
    image_text_result = train_image_text_examples(
        examples=image_text_examples_from_choices(image_examples),
        image_input_layer=image_result.model.image_input_layer,
        text_model=text_artifacts.model,
        tokenizer=text_artifacts.tokenizer,
        prompt=prompt,
        config=image_text_config,
    )
    return ImageTextTrainingPhasesResult(
        image_classification=image_result,
        language_modeling=text_artifacts,
        image_text=image_text_result,
    )


def _model_vocab_size(config: LanguageModelingTrainingConfig) -> int:
    if config.tokenizer == "byte":
        return 257
    return config.tokenizer_vocab_size


def _core_shape() -> dict[str, int | float]:
    preset = TRANSFORMER_CORE_PRESETS["tiny"]
    return {
        "embedding_dim": int(preset["embedding_dim"]),
        "num_heads": int(preset["num_heads"]),
        "hidden_dim": int(preset["hidden_dim"]),
        "num_layers": int(preset["num_layers"]),
        "dropout": float(preset["dropout"]),
    }


def _validate_embedding_dim(image_embedding_dim: int, text_embedding_dim: int) -> None:
    if image_embedding_dim != text_embedding_dim:
        raise ValueError("image classifier and text model embedding dimensions must match")
