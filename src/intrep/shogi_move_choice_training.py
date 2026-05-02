from __future__ import annotations

from dataclasses import dataclass
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
    torch.manual_seed(training_config.seed)
    dataset = ShogiMoveChoiceDataset(examples)
    loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)
    train_eval_loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=False)
    eval_dataset = ShogiMoveChoiceDataset(eval_examples) if eval_examples is not None else None
    eval_loader = (
        DataLoader(eval_dataset, batch_size=training_config.batch_size, shuffle=False)
        if eval_dataset is not None
        else None
    )
    model = build_shogi_move_choice_model(training_config)
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
    while step < training_config.max_steps:
        for position_token_ids, candidate_move_features, candidate_mask, labels, value_targets in loader:
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
    with torch.no_grad():
        for position_token_ids, candidate_move_features, candidate_mask, labels, value_targets in loader:
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
