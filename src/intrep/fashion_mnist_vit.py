from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from intrep.gpt_model import GPT_MODEL_PRESETS
from intrep.gpt_training import resolve_training_device
from intrep.image_io import read_portable_image
from intrep.token_sequence import TokenSequence, token_sequence_from_ids


FASHION_MNIST_LABELS = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class ImageChoiceExample:
    image_path: Path
    choices: tuple[str, ...]
    answer_index: int

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("choices must not be empty")
        if not 0 <= self.answer_index < len(self.choices):
            raise ValueError("answer_index out of range")

    @property
    def answer_text(self) -> str:
        return self.choices[self.answer_index]


@dataclass(frozen=True)
class ImageClassificationConfig:
    patch_size: int = 4
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    seed: int = 7
    model_preset: str = "tiny"
    device: str = "auto"


@dataclass(frozen=True)
class ImageClassificationMetrics:
    target: str
    rendering: str
    train_case_count: int
    eval_case_count: int
    train_initial_loss: float
    train_final_loss: float
    train_accuracy: float
    eval_accuracy: float | None
    patch_size: int
    max_steps: int
    model_preset: str

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "rendering": self.rendering,
            "train_case_count": self.train_case_count,
            "eval_case_count": self.eval_case_count,
            "train_initial_loss": self.train_initial_loss,
            "train_final_loss": self.train_final_loss,
            "train_accuracy": self.train_accuracy,
            "eval_accuracy": self.eval_accuracy,
            "image_patch_size": self.patch_size,
            "max_steps": self.max_steps,
            "model_preset": self.model_preset,
        }


class ImagePatchAdapter(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        height, width = image_size
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("image dimensions must be divisible by patch_size")
        patch_dim = patch_size * patch_size
        patch_count = (height // patch_size) * (width // patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, embedding_dim)
        self.position_embedding = nn.Embedding(patch_count, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 3:
            raise ValueError("images must have shape [batch, height, width]")
        if tuple(images.shape[1:]) != self.image_size:
            raise ValueError("images do not match adapter image_size")
        patches = _patchify(images, self.patch_size)
        positions = torch.arange(patches.size(1), device=images.device).unsqueeze(0)
        return self.patch_embedding(patches) + self.position_embedding(positions)


class SharedTransformerCore(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, sequence, hidden]")
        return self.encoder(embeddings)


class ClassificationHead(nn.Module):
    def __init__(self, *, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden states must have shape [batch, sequence, hidden]")
        pooled = hidden.mean(dim=1)
        return self.output(pooled)


class PatchTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: tuple[int, int],
        patch_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_adapter = ImagePatchAdapter(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
        )
        self.core = SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classification_head = ClassificationHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.core(self.image_adapter(images))
        return self.classification_head(encoded)


def train_fashion_mnist_classifier(
    *,
    train_examples: list[ImageChoiceExample],
    eval_examples: list[ImageChoiceExample] | None = None,
    config: ImageClassificationConfig | None = None,
) -> ImageClassificationMetrics:
    training_config = config or ImageClassificationConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)  # type: ignore[arg-type]
    train_images, train_labels = image_label_tensors_from_examples(train_examples)
    eval_images: torch.Tensor | None = None
    eval_labels: torch.Tensor | None = None
    if eval_examples is not None:
        eval_images, eval_labels = image_label_tensors_from_examples(eval_examples)
        if tuple(eval_images.shape[1:]) != tuple(train_images.shape[1:]):
            raise ValueError("eval images must have the same shape as train images")

    preset = GPT_MODEL_PRESETS[training_config.model_preset]
    model = PatchTransformerClassifier(
        image_size=(int(train_images.shape[1]), int(train_images.shape[2])),
        patch_size=training_config.patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        num_classes=len(FASHION_MNIST_LABELS),
    ).to(device)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    if eval_images is not None and eval_labels is not None:
        eval_images = eval_images.to(device)
        eval_labels = eval_labels.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate)
    initial_loss = _loss(model, loss_fn, train_images, train_labels)
    for step in range(training_config.max_steps):
        start = (step * training_config.batch_size) % len(train_images)
        indices = (torch.arange(training_config.batch_size, device=device) + start) % len(train_images)
        batch_images = train_images.index_select(0, indices)
        batch_labels = train_labels.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_images), batch_labels)
        loss.backward()
        optimizer.step()

    final_loss = _loss(model, loss_fn, train_images, train_labels)
    train_accuracy = _accuracy(model, train_images, train_labels)
    eval_accuracy = None
    eval_count = 0
    if eval_images is not None and eval_labels is not None:
        eval_accuracy = _accuracy(model, eval_images, eval_labels)
        eval_count = int(eval_labels.numel())
    return ImageClassificationMetrics(
        target="label",
        rendering="image-patches",
        train_case_count=int(train_labels.numel()),
        eval_case_count=eval_count,
        train_initial_loss=initial_loss,
        train_final_loss=final_loss,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        patch_size=training_config.patch_size,
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )


def image_label_tensors_from_examples(
    examples: list[ImageChoiceExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for example in examples:
        images.append(_read_image_path(example.image_path))
        labels.append(example.answer_index)
    if not images:
        raise ValueError("examples must not be empty")
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("all images must have the same shape")
    image_tensor = torch.tensor(np.stack(images).astype(np.float32) / 255.0, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor


def load_image_choice_examples_jsonl(path: str | Path) -> list[ImageChoiceExample]:
    examples: list[ImageChoiceExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-choice JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_choice_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-choice JSONL must contain at least one example")
    return examples


def image_choice_example_from_record(record: object, *, line_number: int) -> ImageChoiceExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-choice JSONL at line {line_number}: expected object")
    required = {"image_path", "choices", "answer_index"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-choice JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-choice JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    choices = record["choices"]
    answer_index = record["answer_index"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(
            f"Invalid image-choice JSONL at line {line_number}: image_path must be a string"
        )
    if not isinstance(choices, list) or not all(isinstance(choice, str) for choice in choices):
        raise ValueError(
            f"Invalid image-choice JSONL at line {line_number}: choices must be a list of strings"
        )
    if not isinstance(answer_index, int):
        raise ValueError(f"Invalid image-choice JSONL at line {line_number}: answer_index must be an integer")
    try:
        return ImageChoiceExample(
            image_path=Path(image_path),
            choices=tuple(choices),
            answer_index=answer_index,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-choice JSONL at line {line_number}: {error}") from error


def image_choice_example_to_record(example: ImageChoiceExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "choices": list(example.choices),
        "answer_index": example.answer_index,
    }


def fashion_mnist_label_continuation_sequence(
    example: ImageChoiceExample,
    tokenizer: object,
    *,
    prompt: str = "Class:",
) -> TokenSequence:
    prompt_ids = tokenizer.encode(prompt)
    label_ids = tokenizer.encode(example.answer_text)
    if not label_ids:
        raise ValueError("label text must encode to at least one token")
    image_ids = _image_placeholder_token_ids(example.image_path)
    token_ids = [*image_ids, *prompt_ids, *label_ids]
    loss_mask = [False] * (len(image_ids) + len(prompt_ids)) + [True] * len(label_ids)
    return token_sequence_from_ids(token_ids, loss_mask=loss_mask)


def write_metrics(path: str | Path, metrics: ImageClassificationMetrics) -> None:
    Path(path).write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    return patches.contiguous().view(images.size(0), -1, patch_size * patch_size)


def _read_image_path(path: Path) -> np.ndarray:
    pixels = read_portable_image(path)
    if pixels.ndim == 2:
        return pixels
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return np.rint(pixels.mean(axis=2)).astype(np.uint8)
    raise ValueError("image payload must be grayscale or RGB")


def _image_placeholder_token_ids(path: Path) -> list[int]:
    pixels = _read_image_path(path)
    return [int(value) for value in pixels.reshape(-1).tolist()]


def _validate_config(config: ImageClassificationConfig) -> None:
    if config.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if config.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.model_preset not in GPT_MODEL_PRESETS:
        raise ValueError(f"unknown model preset: {config.model_preset}")


def _loss(
    model: PatchTransformerClassifier,
    loss_fn: nn.CrossEntropyLoss,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    model.eval()
    with torch.no_grad():
        return float(loss_fn(model(images), labels).item())


def _accuracy(model: PatchTransformerClassifier, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1)
    return float((predictions == labels).float().mean().item())
