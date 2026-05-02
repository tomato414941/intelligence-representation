from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from intrep.grid_world import (
    GRID_ACTIONS,
    GridExperienceTransition,
)
from intrep.grid_world_layers import GridObservationInputLayer
from intrep.grid_world_training_data import GridStepPredictionDataset
from intrep.image_training_data import seeded_data_loader
from intrep.language_modeling_training import resolve_training_device
from intrep.training_utils import LearningRateSchedule, build_adamw, build_lr_scheduler, clip_gradients
from intrep.transformer_core import SharedTransformerCore


@dataclass(frozen=True)
class GridStepPredictionConfig:
    max_steps: int = 20
    batch_size: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.01
    max_grad_norm: float | None = 1.0
    lr_schedule: LearningRateSchedule = "constant"
    warmup_steps: int = 0
    seed: int = 7
    embedding_dim: int = 32
    num_heads: int = 2
    hidden_dim: int = 64
    num_layers: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class GridStepPredictionResult:
    train_case_count: int
    eval_case_count: int
    initial_loss: float
    final_loss: float
    final_next_cell_loss: float
    final_reward_loss: float
    final_terminated_loss: float
    next_cell_accuracy: float
    reward_accuracy: float
    terminated_accuracy: float
    max_steps: int
    eval_loss: float | None = None
    eval_next_cell_loss: float | None = None
    eval_reward_loss: float | None = None
    eval_terminated_loss: float | None = None
    eval_next_cell_accuracy: float | None = None
    eval_reward_accuracy: float | None = None
    eval_terminated_accuracy: float | None = None


@dataclass(frozen=True)
class _GridStepPredictionMetrics:
    loss: float
    next_cell_loss: float
    reward_loss: float
    terminated_loss: float
    next_cell_accuracy: float
    reward_accuracy: float
    terminated_accuracy: float


class GridStepPredictor(nn.Module):
    def __init__(
        self,
        *,
        height: int,
        width: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        core: SharedTransformerCore | None = None,
    ) -> None:
        super().__init__()
        self.grid_input = GridObservationInputLayer(height=height, width=width, embedding_dim=embedding_dim)
        self.action_embedding = nn.Embedding(len(GRID_ACTIONS), embedding_dim)
        self.core = core or SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        self.next_cell_output = nn.Linear(embedding_dim, height * width)
        self.reward_output = nn.Linear(embedding_dim, 3)
        self.terminated_output = nn.Linear(embedding_dim, 2)

    def forward(self, observations: torch.Tensor, action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if action_ids.ndim != 1:
            raise ValueError("action_ids must have shape [batch]")
        grid_embeddings = self.grid_input(observations)
        action_embeddings = self.action_embedding(action_ids).unsqueeze(1)
        hidden = self.core(torch.cat((grid_embeddings, action_embeddings), dim=1), causal=False)
        pooled = hidden[:, -1, :]
        return (
            self.next_cell_output(pooled),
            self.reward_output(pooled),
            self.terminated_output(pooled),
        )


def train_grid_step_predictor(
    examples: Sequence[GridExperienceTransition],
    *,
    eval_examples: Sequence[GridExperienceTransition] | None = None,
    config: GridStepPredictionConfig | None = None,
) -> GridStepPredictionResult:
    config = config or GridStepPredictionConfig()
    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    torch.manual_seed(config.seed)
    device = resolve_training_device(config.device)
    dataset = GridStepPredictionDataset(examples)
    eval_dataset = GridStepPredictionDataset(eval_examples) if eval_examples is not None else None
    if eval_dataset is not None and (eval_dataset.height != dataset.height or eval_dataset.width != dataset.width):
        raise ValueError("eval examples must use the same grid size as train examples")
    loader = seeded_data_loader(dataset, batch_size=config.batch_size, seed=config.seed, shuffle=True, device=device)
    train_eval_loader = seeded_data_loader(dataset, batch_size=config.batch_size, seed=config.seed, shuffle=False, device=device)
    eval_loader = (
        seeded_data_loader(eval_dataset, batch_size=config.batch_size, seed=config.seed, shuffle=False, device=device)
        if eval_dataset is not None
        else None
    )
    model = GridStepPredictor(
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

    initial_metrics = _evaluate(model, train_eval_loader, device)
    iterator = iter(loader)
    for _ in range(config.max_steps):
        try:
            observations, action_ids, next_cell_targets, reward_targets, terminated_targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            observations, action_ids, next_cell_targets, reward_targets, terminated_targets = next(iterator)
        observations = observations.to(device)
        action_ids = action_ids.to(device)
        next_cell_targets = next_cell_targets.to(device)
        reward_targets = reward_targets.to(device)
        terminated_targets = terminated_targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        next_cell_logits, reward_logits, terminated_logits = model(observations, action_ids)
        loss = (
            nn.functional.cross_entropy(next_cell_logits, next_cell_targets)
            + nn.functional.cross_entropy(reward_logits, reward_targets)
            + nn.functional.cross_entropy(terminated_logits, terminated_targets)
        )
        loss.backward()
        clip_gradients(model, config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    train_metrics = _evaluate(model, train_eval_loader, device)
    held_out_metrics = _evaluate(model, eval_loader, device) if eval_loader is not None else None
    return GridStepPredictionResult(
        train_case_count=len(dataset),
        eval_case_count=len(eval_dataset) if eval_dataset is not None else 0,
        initial_loss=initial_metrics.loss,
        final_loss=train_metrics.loss,
        final_next_cell_loss=train_metrics.next_cell_loss,
        final_reward_loss=train_metrics.reward_loss,
        final_terminated_loss=train_metrics.terminated_loss,
        next_cell_accuracy=train_metrics.next_cell_accuracy,
        reward_accuracy=train_metrics.reward_accuracy,
        terminated_accuracy=train_metrics.terminated_accuracy,
        eval_loss=held_out_metrics.loss if held_out_metrics is not None else None,
        eval_next_cell_loss=held_out_metrics.next_cell_loss if held_out_metrics is not None else None,
        eval_reward_loss=held_out_metrics.reward_loss if held_out_metrics is not None else None,
        eval_terminated_loss=held_out_metrics.terminated_loss if held_out_metrics is not None else None,
        eval_next_cell_accuracy=held_out_metrics.next_cell_accuracy if held_out_metrics is not None else None,
        eval_reward_accuracy=held_out_metrics.reward_accuracy if held_out_metrics is not None else None,
        eval_terminated_accuracy=held_out_metrics.terminated_accuracy if held_out_metrics is not None else None,
        max_steps=config.max_steps,
    )


def _evaluate(
    model: GridStepPredictor,
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> _GridStepPredictionMetrics:
    model.eval()
    total_next_cell_loss = 0.0
    total_reward_loss = 0.0
    total_terminated_loss = 0.0
    total_next_cell_correct = 0
    total_reward_correct = 0
    total_terminated_correct = 0
    total_count = 0
    with torch.no_grad():
        for observations, action_ids, next_cell_targets, reward_targets, terminated_targets in data_loader:
            observations = observations.to(device)
            action_ids = action_ids.to(device)
            next_cell_targets = next_cell_targets.to(device)
            reward_targets = reward_targets.to(device)
            terminated_targets = terminated_targets.to(device)
            next_cell_logits, reward_logits, terminated_logits = model(observations, action_ids)
            next_cell_loss = nn.functional.cross_entropy(next_cell_logits, next_cell_targets, reduction="sum")
            reward_loss = nn.functional.cross_entropy(reward_logits, reward_targets, reduction="sum")
            terminated_loss = nn.functional.cross_entropy(terminated_logits, terminated_targets, reduction="sum")
            total_next_cell_loss += float(next_cell_loss.item())
            total_reward_loss += float(reward_loss.item())
            total_terminated_loss += float(terminated_loss.item())
            total_next_cell_correct += int((next_cell_logits.argmax(dim=1) == next_cell_targets).sum().item())
            total_reward_correct += int((reward_logits.argmax(dim=1) == reward_targets).sum().item())
            total_terminated_correct += int((terminated_logits.argmax(dim=1) == terminated_targets).sum().item())
            total_count += int(next_cell_targets.numel())
    model.train()
    next_cell_loss = total_next_cell_loss / total_count
    reward_loss = total_reward_loss / total_count
    terminated_loss = total_terminated_loss / total_count
    return _GridStepPredictionMetrics(
        loss=next_cell_loss + reward_loss + terminated_loss,
        next_cell_loss=next_cell_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
        next_cell_accuracy=total_next_cell_correct / total_count,
        reward_accuracy=total_reward_correct / total_count,
        terminated_accuracy=total_terminated_correct / total_count,
    )
