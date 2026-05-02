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


@dataclass(frozen=True)
class ShogiMoveChoiceTrainingMetrics:
    train_case_count: int
    initial_loss: float
    final_loss: float
    accuracy: float
    max_steps: int


@dataclass(frozen=True)
class ShogiMoveChoiceTrainingResult:
    model: nn.Module
    config: ShogiMoveChoiceTrainingConfig
    metrics: ShogiMoveChoiceTrainingMetrics


def train_shogi_move_choice_model(
    examples: Sequence[ShogiMoveChoiceExample],
    *,
    config: ShogiMoveChoiceTrainingConfig | None = None,
) -> ShogiMoveChoiceTrainingResult:
    training_config = config or ShogiMoveChoiceTrainingConfig()
    if training_config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    torch.manual_seed(training_config.seed)
    dataset = ShogiMoveChoiceDataset(examples)
    loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=training_config.batch_size, shuffle=False)
    model = build_shogi_move_choice_model(training_config)
    optimizer = build_adamw(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    initial_loss, _ = evaluate_shogi_move_choice_model(model, eval_loader)

    model.train()
    step = 0
    while step < training_config.max_steps:
        for position_token_ids, candidate_move_features, candidate_mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(position_token_ids, candidate_move_features, candidate_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= training_config.max_steps:
                break

    final_loss, accuracy = evaluate_shogi_move_choice_model(model, eval_loader)
    return ShogiMoveChoiceTrainingResult(
        model=model,
        config=training_config,
        metrics=ShogiMoveChoiceTrainingMetrics(
            train_case_count=len(dataset),
            initial_loss=initial_loss,
            final_loss=final_loss,
            accuracy=accuracy,
            max_steps=training_config.max_steps,
        ),
    )


def evaluate_shogi_move_choice_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for position_token_ids, candidate_move_features, candidate_mask, labels in loader:
            logits = model(position_token_ids, candidate_move_features, candidate_mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            losses.append(float(loss.item()))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return sum(losses) / len(losses), correct / total


def train_shogi_move_choice_model_from_usi_file(
    path: str,
    *,
    config: ShogiMoveChoiceTrainingConfig | None = None,
) -> ShogiMoveChoiceTrainingResult:
    return train_shogi_move_choice_model(
        load_shogi_move_choice_examples_from_usi_file(path),
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
