from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from intrep.shogi_game_record import load_shogi_move_choice_examples_from_usi_file
from intrep.shogi_move_choice import ShogiMoveChoiceDataset, ShogiMoveChoiceExample
from intrep.shogi_move_choice_model import (
    SharedCoreShogiMoveChoiceModel,
    SharedCoreShogiMoveChoiceModelConfig,
    ShogiMoveChoiceModel,
    ShogiMoveChoiceModelConfig,
)
from intrep.training_utils import build_adamw


@dataclass(frozen=True)
class ShogiMoveChoiceTrainingConfig:
    max_steps: int = 100
    batch_size: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    seed: int = 7
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    use_shared_core: bool = True
    value_loss_weight: float = 0.0
    device: str = "cpu"
    max_train_eval_examples: int | None = None
    max_eval_examples: int | None = None
    log_every: int | None = None
    num_workers: int = 0
    pin_memory: bool = False


@dataclass(frozen=True)
class ShogiMoveChoiceEvaluationMetrics:
    loss: float
    accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_reciprocal_rank: float
    mean_correct_move_rank: float
    value_loss: float | None = None


@dataclass(frozen=True)
class ShogiMoveChoiceTrainingMetrics:
    train_case_count: int
    eval_case_count: int
    initial_loss: float
    initial_value_loss: float | None
    final_loss: float
    accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_reciprocal_rank: float
    mean_correct_move_rank: float
    value_loss: float | None
    eval_loss: float | None
    initial_eval_loss: float | None
    eval_accuracy: float | None
    initial_eval_accuracy: float | None
    eval_top_3_accuracy: float | None
    eval_top_5_accuracy: float | None
    eval_mean_reciprocal_rank: float | None
    eval_mean_correct_move_rank: float | None
    eval_value_loss: float | None
    initial_eval_value_loss: float | None
    max_steps: int


@dataclass(frozen=True)
class ShogiMoveChoiceTrainingResult:
    model: nn.Module
    config: ShogiMoveChoiceTrainingConfig
    metrics: ShogiMoveChoiceTrainingMetrics


def train_shogi_move_choice_model(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    eval_examples: Sequence[ShogiMoveChoiceExample] | None = None,
    config: ShogiMoveChoiceTrainingConfig | None = None,
) -> ShogiMoveChoiceTrainingResult:
    training_config = config or ShogiMoveChoiceTrainingConfig()
    if training_config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if training_config.log_every is not None and training_config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if training_config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)
    dataset = ShogiMoveChoiceDataset(examples)
    loader = _build_shogi_move_choice_loader(dataset, training_config, shuffle=True)
    train_eval_examples = _limit_examples(examples, training_config.max_train_eval_examples)
    train_eval_dataset = ShogiMoveChoiceDataset(train_eval_examples)
    train_eval_loader = _build_shogi_move_choice_loader(train_eval_dataset, training_config, shuffle=False)
    limited_eval_examples = (
        _limit_examples(eval_examples, training_config.max_eval_examples) if eval_examples is not None else None
    )
    eval_dataset = ShogiMoveChoiceDataset(limited_eval_examples) if limited_eval_examples is not None else None
    eval_loader = (
        _build_shogi_move_choice_loader(eval_dataset, training_config, shuffle=False)
        if eval_dataset is not None
        else None
    )
    model = build_shogi_move_choice_model(training_config).to(device)
    optimizer = build_adamw(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    initial_metrics = evaluate_shogi_move_choice_metrics(model, train_eval_loader)
    initial_eval_metrics: ShogiMoveChoiceEvaluationMetrics | None = None
    if eval_loader is not None:
        initial_eval_metrics = evaluate_shogi_move_choice_metrics(model, eval_loader)

    model.train()
    step = 0
    started = time.monotonic()
    while step < training_config.max_steps:
        for position_token_ids, candidate_move_features, candidate_mask, labels, value_targets in loader:
            position_token_ids = position_token_ids.to(device)
            candidate_move_features = candidate_move_features.to(device)
            candidate_mask = candidate_mask.to(device)
            labels = labels.to(device)
            value_targets = value_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(position_token_ids, candidate_move_features, candidate_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            value_mask = torch.isfinite(value_targets)
            if training_config.value_loss_weight > 0.0 and value_mask.any():
                value_predictions = model.predict_value(position_token_ids)
                value_loss = torch.nn.functional.mse_loss(value_predictions[value_mask], value_targets[value_mask])
                loss = loss + training_config.value_loss_weight * value_loss
            loss.backward()
            optimizer.step()
            step += 1
            if training_config.log_every is not None and step % training_config.log_every == 0:
                _log_training_progress(step, training_config.max_steps, started, loss, device)
            if step >= training_config.max_steps:
                break

    final_metrics = evaluate_shogi_move_choice_metrics(model, train_eval_loader)
    eval_metrics: ShogiMoveChoiceEvaluationMetrics | None = None
    if eval_loader is not None:
        eval_metrics = evaluate_shogi_move_choice_metrics(model, eval_loader)
    return ShogiMoveChoiceTrainingResult(
        model=model,
        config=training_config,
        metrics=ShogiMoveChoiceTrainingMetrics(
            train_case_count=len(dataset),
            eval_case_count=len(eval_dataset) if eval_dataset is not None else 0,
            initial_loss=initial_metrics.loss,
            initial_value_loss=initial_metrics.value_loss,
            final_loss=final_metrics.loss,
            accuracy=final_metrics.accuracy,
            top_3_accuracy=final_metrics.top_3_accuracy,
            top_5_accuracy=final_metrics.top_5_accuracy,
            mean_reciprocal_rank=final_metrics.mean_reciprocal_rank,
            mean_correct_move_rank=final_metrics.mean_correct_move_rank,
            value_loss=final_metrics.value_loss,
            eval_loss=eval_metrics.loss if eval_metrics is not None else None,
            initial_eval_loss=initial_eval_metrics.loss if initial_eval_metrics is not None else None,
            eval_accuracy=eval_metrics.accuracy if eval_metrics is not None else None,
            initial_eval_accuracy=initial_eval_metrics.accuracy if initial_eval_metrics is not None else None,
            eval_top_3_accuracy=eval_metrics.top_3_accuracy if eval_metrics is not None else None,
            eval_top_5_accuracy=eval_metrics.top_5_accuracy if eval_metrics is not None else None,
            eval_mean_reciprocal_rank=eval_metrics.mean_reciprocal_rank if eval_metrics is not None else None,
            eval_mean_correct_move_rank=eval_metrics.mean_correct_move_rank if eval_metrics is not None else None,
            eval_value_loss=eval_metrics.value_loss if eval_metrics is not None else None,
            initial_eval_value_loss=initial_eval_metrics.value_loss if initial_eval_metrics is not None else None,
            max_steps=training_config.max_steps,
        ),
    )


def evaluate_shogi_move_choice_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    metrics = evaluate_shogi_move_choice_metrics(model, loader)
    return metrics.loss, metrics.accuracy


def evaluate_shogi_move_choice_metrics(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> ShogiMoveChoiceEvaluationMetrics:
    model.eval()
    losses: list[float] = []
    value_losses: list[float] = []
    correct = 0
    top_3_correct = 0
    top_5_correct = 0
    reciprocal_rank_sum = 0.0
    rank_sum = 0.0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for position_token_ids, candidate_move_features, candidate_mask, labels, value_targets in loader:
            position_token_ids = position_token_ids.to(device)
            candidate_move_features = candidate_move_features.to(device)
            candidate_mask = candidate_mask.to(device)
            labels = labels.to(device)
            value_targets = value_targets.to(device)
            logits = model(position_token_ids, candidate_move_features, candidate_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            losses.append(float(loss.item()))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            sorted_indices = logits.argsort(dim=1, descending=True)
            label_matches = sorted_indices.eq(labels[:, None])
            ranks = label_matches.float().argmax(dim=1) + 1
            top_3_correct += int((ranks <= 3).sum().item())
            top_5_correct += int((ranks <= 5).sum().item())
            reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
            rank_sum += float(ranks.float().sum().item())
            total += int(labels.numel())
            value_mask = torch.isfinite(value_targets)
            if value_mask.any() and hasattr(model, "predict_value"):
                value_predictions = model.predict_value(position_token_ids)
                value_loss = torch.nn.functional.mse_loss(value_predictions[value_mask], value_targets[value_mask])
                value_losses.append(float(value_loss.item()))
    return ShogiMoveChoiceEvaluationMetrics(
        loss=sum(losses) / len(losses),
        accuracy=correct / total,
        top_3_accuracy=top_3_correct / total,
        top_5_accuracy=top_5_correct / total,
        mean_reciprocal_rank=reciprocal_rank_sum / total,
        mean_correct_move_rank=rank_sum / total,
        value_loss=sum(value_losses) / len(value_losses) if value_losses else None,
    )


def _build_shogi_move_choice_loader(
    dataset: ShogiMoveChoiceDataset,
    config: ShogiMoveChoiceTrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _limit_examples(
    examples: Sequence[ShogiMoveChoiceExample],
    max_examples: int | None,
) -> Sequence[ShogiMoveChoiceExample]:
    if max_examples is None:
        return examples
    if max_examples <= 0:
        raise ValueError("max eval examples must be positive")
    return examples[:max_examples]


def _log_training_progress(
    step: int,
    max_steps: int,
    started: float,
    loss: torch.Tensor,
    device: torch.device,
) -> None:
    elapsed = time.monotonic() - started
    steps_per_second = step / elapsed if elapsed > 0.0 else 0.0
    parts = [
        f"step={step}/{max_steps}",
        f"elapsed_seconds={elapsed:.1f}",
        f"steps_per_second={steps_per_second:.3f}",
        f"loss={float(loss.detach().item()):.4f}",
        f"device={device}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        parts.append(f"cuda_max_memory_mb={allocated_mb:.1f}")
    print(" ".join(parts), flush=True)


def train_shogi_move_choice_model_from_usi_file(
    path: str,
    *,
    eval_path: str | None = None,
    config: ShogiMoveChoiceTrainingConfig | None = None,
) -> ShogiMoveChoiceTrainingResult:
    return train_shogi_move_choice_model(
        load_shogi_move_choice_examples_from_usi_file(path),
        eval_examples=load_shogi_move_choice_examples_from_usi_file(eval_path) if eval_path is not None else None,
        config=config,
    )


def build_shogi_move_choice_model(config: ShogiMoveChoiceTrainingConfig) -> nn.Module:
    if config.use_shared_core:
        return SharedCoreShogiMoveChoiceModel(
            SharedCoreShogiMoveChoiceModelConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        )
    return ShogiMoveChoiceModel(
        ShogiMoveChoiceModelConfig(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        )
    )
