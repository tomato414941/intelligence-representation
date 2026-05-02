from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from intrep.image_input_layer import ImagePatchInputLayer
from intrep.language_modeling_training import resolve_training_device
from intrep.image_io import read_portable_image
from intrep.image_training_data import (
    channel_count_from_image_shape,
    image_tensor_from_path,
    seeded_data_loader,
)
from intrep.model_presets import TRANSFORMER_CORE_PRESETS
from intrep.shared_multimodal_model import ClassificationHead, SharedMultimodalModel
from intrep.training_utils import LearningRateSchedule, build_adamw, build_lr_scheduler, clip_gradients


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

MNIST_LABELS = tuple(str(index) for index in range(10))

CIFAR10_LABELS = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class ImageClassificationExample:
    image_path: Path
    label_names: tuple[str, ...]
    label_index: int

    def __post_init__(self) -> None:
        if not self.label_names:
            raise ValueError("label_names must not be empty")
        if not 0 <= self.label_index < len(self.label_names):
            raise ValueError("label_index out of range")

    @property
    def label_text(self) -> str:
        return self.label_names[self.label_index]


@dataclass(frozen=True)
class ImageClassificationConfig:
    patch_size: int = 4
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    seed: int = 7
    model_preset: str = "tiny"
    device: str = "auto"


@dataclass(frozen=True)
class ImageClassificationMetrics:
    target: str
    input_representation: str
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
            "input_representation": self.input_representation,
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


@dataclass(frozen=True)
class ImageClassificationTrainingResult:
    metrics: ImageClassificationMetrics
    model: SharedMultimodalModel
    config: ImageClassificationConfig
    image_shape: tuple[int, ...]
    label_names: tuple[str, ...]


class ImageClassificationDatasetLike(Protocol):
    image_shape: tuple[int, ...]
    channel_count: int
    label_names: tuple[str, ...]

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]: ...


class ImageClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[ImageClassificationExample]) -> None:
        if not examples:
            raise ValueError("examples must not be empty")
        _class_count_from_examples(examples)
        self.examples = tuple(examples)
        self.label_names = examples[0].label_names
        self.image_shape = tuple(int(value) for value in image_tensor_from_path(examples[0].image_path).shape)
        self.channel_count = channel_count_from_image_shape(self.image_shape)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        image = image_tensor_from_path(example.image_path)
        if tuple(image.shape) != self.image_shape:
            raise ValueError("all images must have the same shape")
        label = torch.tensor(example.label_index, dtype=torch.long)
        return image, label


class ImageFolderClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        root: str | Path,
        *,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        try:
            from PIL import Image
            from torchvision.datasets import ImageFolder
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "ImageFolderClassificationDataset requires the vision extra. "
                "Install with `uv sync --extra torch --extra vision`."
            ) from error

        self._image_module = Image
        self._dataset = ImageFolder(str(root))
        if len(self._dataset) == 0:
            raise ValueError("image folder root must contain image files")
        self.label_names = tuple(self._dataset.classes)
        first_image, _ = self._dataset[0]
        first_rgb = first_image.convert("RGB")
        if image_size is None:
            width, height = first_rgb.size
            self.image_size = (height, width)
        else:
            self.image_size = image_size
        self.image_shape = (self.image_size[0], self.image_size[1], 3)
        self.channel_count = 3

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label_index = self._dataset[index]
        image = image.convert("RGB")
        if (image.height, image.width) != self.image_size:
            image = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=self._image_module.Resampling.BILINEAR,
            )
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        return torch.tensor(pixels, dtype=torch.float32), torch.tensor(label_index, dtype=torch.long)


def train_image_classifier(
    *,
    train_examples: list[ImageClassificationExample],
    eval_examples: list[ImageClassificationExample] | None = None,
    config: ImageClassificationConfig | None = None,
) -> ImageClassificationMetrics:
    return train_image_classifier_with_result(
        train_examples=train_examples,
        eval_examples=eval_examples,
        config=config,
    ).metrics


def train_image_classifier_with_result(
    *,
    train_examples: list[ImageClassificationExample] | None = None,
    eval_examples: list[ImageClassificationExample] | None = None,
    train_dataset: ImageClassificationDatasetLike | None = None,
    eval_dataset: ImageClassificationDatasetLike | None = None,
    config: ImageClassificationConfig | None = None,
) -> ImageClassificationTrainingResult:
    training_config = config or ImageClassificationConfig()
    _validate_config(training_config)
    torch.manual_seed(training_config.seed)
    device = resolve_training_device(training_config.device)  # type: ignore[arg-type]
    if train_dataset is None:
        if train_examples is None:
            raise ValueError("train_examples or train_dataset is required")
        train_dataset = ImageClassificationDataset(train_examples)
    elif train_examples is not None:
        raise ValueError("provide only one of train_examples or train_dataset")
    train_loader = _image_classification_data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=True,
        device=device,
    )
    train_eval_loader = _image_classification_data_loader(
        train_dataset,
        batch_size=training_config.batch_size,
        seed=training_config.seed,
        shuffle=False,
        device=device,
    )
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if eval_examples is not None:
        if eval_dataset is not None:
            raise ValueError("provide only one of eval_examples or eval_dataset")
        eval_dataset = ImageClassificationDataset(eval_examples)
        if eval_dataset.label_names != train_dataset.label_names:
            raise ValueError("eval examples must use the same label_names as train examples")
    if eval_dataset is not None:
        if eval_dataset.image_shape != train_dataset.image_shape:
            raise ValueError("eval images must have the same shape as train images")
        if eval_dataset.label_names != train_dataset.label_names:
            raise ValueError("eval examples must use the same label_names as train examples")
        eval_loader = _image_classification_data_loader(
            eval_dataset,
            batch_size=training_config.batch_size,
            seed=training_config.seed,
            shuffle=False,
            device=device,
        )

    preset = TRANSFORMER_CORE_PRESETS[training_config.model_preset]
    model = SharedMultimodalModel(
        vocab_size=1,
        text_context_length=1,
        image_size=(train_dataset.image_shape[0], train_dataset.image_shape[1]),
        patch_size=training_config.patch_size,
        embedding_dim=int(preset["embedding_dim"]),
        num_heads=int(preset["num_heads"]),
        hidden_dim=int(preset["hidden_dim"]),
        num_layers=int(preset["num_layers"]),
        dropout=float(preset["dropout"]),
        channel_count=train_dataset.channel_count,
        num_classes=len(train_dataset.label_names),
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_adamw(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=training_config.lr_schedule,
        warmup_steps=training_config.warmup_steps,
        max_steps=training_config.max_steps,
    )
    initial_loss = _loss(model, loss_fn, train_eval_loader, device)
    train_iterator = iter(train_loader)
    model.train()
    for step in range(training_config.max_steps):
        try:
            batch_images, batch_labels = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch_images, batch_labels = next(train_iterator)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model.image_classification_logits(batch_images), batch_labels)
        loss.backward()
        clip_gradients(model, training_config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    final_loss = _loss(model, loss_fn, train_eval_loader, device)
    train_accuracy = _accuracy(model, train_eval_loader, device)
    eval_accuracy = None
    eval_count = 0
    if eval_dataset is not None and eval_loader is not None:
        eval_accuracy = _accuracy(model, eval_loader, device)
        eval_count = len(eval_dataset)
    metrics = ImageClassificationMetrics(
        target="label",
        input_representation="image-patches",
        train_case_count=len(train_dataset),
        eval_case_count=eval_count,
        train_initial_loss=initial_loss,
        train_final_loss=final_loss,
        train_accuracy=train_accuracy,
        eval_accuracy=eval_accuracy,
        patch_size=training_config.patch_size,
        max_steps=training_config.max_steps,
        model_preset=training_config.model_preset,
    )
    return ImageClassificationTrainingResult(
        metrics=metrics,
        model=model,
        config=training_config,
        image_shape=train_dataset.image_shape,
        label_names=train_dataset.label_names,
    )


def image_classification_tensors_from_examples(
    examples: list[ImageClassificationExample],
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for example in examples:
        images.append(_read_image_path(example.image_path))
        labels.append(example.label_index)
    if not images:
        raise ValueError("examples must not be empty")
    first_shape = images[0].shape
    if any(image.shape != first_shape for image in images):
        raise ValueError("all images must have the same shape")
    image_tensor = torch.tensor(np.stack(images).astype(np.float32) / 255.0, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return image_tensor, label_tensor


def load_image_classification_examples_jsonl(path: str | Path) -> list[ImageClassificationExample]:
    examples: list[ImageClassificationExample] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid image-classification JSONL at line {line_number}: {error.msg}") from error
        examples.append(image_classification_example_from_record(record, line_number=line_number))
    if not examples:
        raise ValueError("image-classification JSONL must contain at least one example")
    return examples


def image_classification_example_from_record(
    record: object,
    *,
    line_number: int,
) -> ImageClassificationExample:
    if not isinstance(record, dict):
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: expected object")
    required = {"image_path", "label_names", "label_index"}
    missing = required - record.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: missing fields: {fields}")
    extra = set(record.keys()) - required
    if extra:
        fields = ", ".join(sorted(extra))
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: unsupported fields: {fields}")
    image_path = record["image_path"]
    label_names = record["label_names"]
    label_index = record["label_index"]
    if not isinstance(image_path, str) or not image_path:
        raise ValueError(
            f"Invalid image-classification JSONL at line {line_number}: image_path must be a string"
        )
    if not isinstance(label_names, list) or not all(isinstance(label_name, str) for label_name in label_names):
        raise ValueError(
            f"Invalid image-classification JSONL at line {line_number}: label_names must be a list of strings"
        )
    if not isinstance(label_index, int):
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: label_index must be an integer")
    try:
        return ImageClassificationExample(
            image_path=Path(image_path),
            label_names=tuple(label_names),
            label_index=label_index,
        )
    except ValueError as error:
        raise ValueError(f"Invalid image-classification JSONL at line {line_number}: {error}") from error


def image_classification_example_to_record(example: ImageClassificationExample) -> dict[str, object]:
    return {
        "image_path": str(example.image_path),
        "label_names": list(example.label_names),
        "label_index": example.label_index,
    }


def _class_count_from_examples(examples: list[ImageClassificationExample]) -> int:
    if not examples:
        raise ValueError("examples must not be empty")
    label_names = examples[0].label_names
    for example in examples:
        if example.label_names != label_names:
            raise ValueError("all examples must use the same label_names")
    return len(label_names)


def write_metrics(path: str | Path, metrics: ImageClassificationMetrics) -> None:
    Path(path).write_text(json.dumps(metrics.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_image_path(path: Path) -> np.ndarray:
    pixels = read_portable_image(path)
    if pixels.ndim == 2:
        return pixels
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return pixels
    raise ValueError("image payload must be grayscale or RGB")


def _image_classification_data_loader(
    dataset: ImageClassificationDatasetLike,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return seeded_data_loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        device=device,
    )


def _validate_config(config: ImageClassificationConfig) -> None:
    if config.patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if config.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if config.lr_schedule not in ("constant", "warmup_cosine"):
        raise ValueError("lr_schedule must be one of: constant, warmup_cosine")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.model_preset not in TRANSFORMER_CORE_PRESETS:
        raise ValueError(f"unknown model preset: {config.model_preset}")


def _loss(
    model: SharedMultimodalModel,
    loss_fn: nn.CrossEntropyLoss,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = loss_fn(model.image_classification_logits(images), labels)
            total_loss += float(loss.item()) * int(labels.numel())
            sample_count += int(labels.numel())
    if was_training:
        model.train()
    return total_loss / sample_count


def _accuracy(
    model: SharedMultimodalModel,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model.image_classification_logits(images).argmax(dim=1)
            correct_count += int((predictions == labels).sum().item())
            total_count += int(labels.numel())
    if was_training:
        model.train()
    return correct_count / total_count
