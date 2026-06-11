from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from intrep.core.training_utils import (
    LearningRateSchedule,
    build_adamw,
    build_lr_scheduler,
    clip_gradients,
    resolve_training_device,
    seeded_data_loader,
)
from intrep.problems.cellular_step_prediction.dataset import CellularStepPredictionDataset
from intrep.problems.cellular_step_prediction.metrics import (
    CellularStepPredictionScores,
    cellular_step_prediction_scores,
)
from intrep.representation.assemblies.cellular_step_prediction import CellularStepPredictionModel
from intrep.worlds.cellular.world import CellularTransition


@dataclass(frozen=True)
class CellularStepPredictionConfig:
    max_steps: int = 1000
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_schedule: LearningRateSchedule = "warmup_cosine"
    warmup_steps: int = 100
    seed: int = 31
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    device: str = "auto"


@dataclass(frozen=True)
class CellularStepTrainingResult:
    train_case_count: int
    initial_loss: float
    final_loss: float
    train_scores: CellularStepPredictionScores
    max_steps: int


@dataclass(frozen=True)
class CellularStepTrainingArtifacts:
    result: CellularStepTrainingResult
    model: CellularStepPredictionModel
    config: CellularStepPredictionConfig
    grid_size: tuple[int, int]


def train_cellular_step_predictor(
    transitions: Sequence[CellularTransition],
    *,
    config: CellularStepPredictionConfig | None = None,
) -> CellularStepTrainingArtifacts:
    config = config or CellularStepPredictionConfig()
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    dataset = CellularStepPredictionDataset(transitions)
    loader = seeded_data_loader(dataset, batch_size=config.batch_size, seed=config.seed, shuffle=True, device=device)
    model = CellularStepPredictionModel(
        height=dataset.height,
        width=dataset.width,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    ).to(device)
    optimizer = build_adamw(model, learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=config.lr_schedule,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
    )

    initial_loss = _mean_loss(model, dataset, device, batch_size=config.batch_size)
    iterator = iter(loader)
    for _ in range(config.max_steps):
        try:
            observations, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            observations, targets = next(iterator)
        observations = observations.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(observations)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        loss.backward()
        clip_gradients(model, config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    final_loss = _mean_loss(model, dataset, device, batch_size=config.batch_size)
    train_scores = cellular_step_prediction_scores(
        predict_cell_ids(model, dataset, device, batch_size=config.batch_size),
        dataset.transitions,
    )
    return CellularStepTrainingArtifacts(
        result=CellularStepTrainingResult(
            train_case_count=len(dataset),
            initial_loss=initial_loss,
            final_loss=final_loss,
            train_scores=train_scores,
            max_steps=config.max_steps,
        ),
        model=model,
        config=config,
        grid_size=(dataset.height, dataset.width),
    )


def predict_cell_ids(
    model: CellularStepPredictionModel,
    dataset: CellularStepPredictionDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            observations = torch.stack(
                [dataset[index][0] for index in range(start, min(start + batch_size, len(dataset)))]
            ).to(device)
            predictions.append(model(observations).argmax(dim=2).cpu())
    model.train()
    return torch.cat(predictions)


def _mean_loss(
    model: CellularStepPredictionModel,
    dataset: CellularStepPredictionDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_cells = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            observations = torch.stack([item[0] for item in batch]).to(device)
            targets = torch.stack([item[1] for item in batch]).to(device)
            logits = model(observations)
            total_loss += float(
                nn.functional.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1), reduction="sum").item()
            )
            total_cells += int(targets.numel())
    model.train()
    return total_loss / total_cells
