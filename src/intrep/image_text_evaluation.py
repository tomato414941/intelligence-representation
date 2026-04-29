from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intrep.causal_text_model import CausalTextModel
from intrep.fashion_mnist_vit import ImageChoiceExample, ImagePatchInputLayer, image_label_tensors_from_examples
from intrep.image_text_scoring import score_image_text_candidates
from intrep.text_tokenizer import TextTokenizer


@dataclass(frozen=True)
class ImageTextChoiceMetrics:
    case_count: int
    accuracy: float
    predicted_indices: tuple[int, ...]
    losses: tuple[tuple[float, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_count": self.case_count,
            "accuracy": self.accuracy,
            "predicted_indices": list(self.predicted_indices),
            "losses": [list(row) for row in self.losses],
        }


def evaluate_image_text_choices(
    *,
    examples: Sequence[ImageChoiceExample],
    image_input_layer: ImagePatchInputLayer,
    text_model: CausalTextModel,
    tokenizer: TextTokenizer,
    prompt: str,
) -> ImageTextChoiceMetrics:
    if not examples:
        raise ValueError("examples must not be empty")
    images, labels = image_label_tensors_from_examples(list(examples))
    predicted_indices: list[int] = []
    loss_rows: list[tuple[float, ...]] = []
    for image, example in zip(images, examples, strict=True):
        losses = score_image_text_candidates(
            image_input_layer=image_input_layer,
            text_model=text_model,
            tokenizer=tokenizer,
            image=image,
            prompt=prompt,
            candidates=example.choices,
        )
        loss_rows.append(tuple(losses))
        predicted_indices.append(min(range(len(losses)), key=losses.__getitem__))

    correct_count = sum(
        int(predicted_index == int(label.item()))
        for predicted_index, label in zip(predicted_indices, labels, strict=True)
    )
    return ImageTextChoiceMetrics(
        case_count=len(examples),
        accuracy=correct_count / len(examples),
        predicted_indices=tuple(predicted_indices),
        losses=tuple(loss_rows),
    )
