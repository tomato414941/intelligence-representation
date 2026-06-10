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
from intrep.domains.grid.world import GridExperienceTransition
from intrep.problems.grid_step_prediction.dataset import GridStepPredictionDataset
from intrep.problems.grid_step_prediction.metrics import NextObservationMetrics, next_observation_metrics
from intrep.representation.assemblies.grid_step_prediction import GridStepPredictionModel

_AGENT_CLASS_ID = 1


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
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    device: str = "auto"


@dataclass(frozen=True)
class GridStepPredictionResult:
    train_case_count: int
    eval_case_count: int
    initial_loss: float
    final_loss: float
    final_next_observation_loss: float
    final_reward_loss: float
    final_terminated_loss: float
    per_cell_accuracy: float
    changed_cell_accuracy: float | None
    whole_grid_match: float
    next_agent_cell_accuracy: float
    reward_accuracy: float
    terminated_accuracy: float
    max_steps: int
    eval_loss: float | None = None
    eval_next_observation_loss: float | None = None
    eval_reward_loss: float | None = None
    eval_terminated_loss: float | None = None
    eval_per_cell_accuracy: float | None = None
    eval_changed_cell_accuracy: float | None = None
    eval_whole_grid_match: float | None = None
    eval_next_agent_cell_accuracy: float | None = None
    eval_reward_accuracy: float | None = None
    eval_terminated_accuracy: float | None = None


@dataclass(frozen=True)
class GridStepTrainingArtifacts:
    result: GridStepPredictionResult
    model: GridStepPredictionModel
    config: GridStepPredictionConfig
    grid_size: tuple[int, int]


@dataclass(frozen=True)
class _SplitMetrics:
    loss: float
    next_observation_loss: float
    reward_loss: float
    terminated_loss: float
    observation: NextObservationMetrics
    reward_accuracy: float
    terminated_accuracy: float


def train_grid_step_predictor(
    examples: Sequence[GridExperienceTransition],
    *,
    eval_examples: Sequence[GridExperienceTransition] | None = None,
    config: GridStepPredictionConfig | None = None,
) -> GridStepPredictionResult:
    return train_grid_step_predictor_with_artifacts(
        examples,
        eval_examples=eval_examples,
        config=config,
    ).result


def train_grid_step_predictor_with_artifacts(
    examples: Sequence[GridExperienceTransition],
    *,
    eval_examples: Sequence[GridExperienceTransition] | None = None,
    config: GridStepPredictionConfig | None = None,
) -> GridStepTrainingArtifacts:
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
    model = GridStepPredictionModel(
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

    initial_metrics = _evaluate(model, dataset, device, batch_size=config.batch_size)
    iterator = iter(loader)
    for _ in range(config.max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        observations, action_ids, next_observation_targets, reward_targets, terminated_targets = (
            tensor.to(device) for tensor in batch
        )
        optimizer.zero_grad(set_to_none=True)
        next_observation_logits, reward_logits, terminated_logits = model(observations, action_ids)
        loss = (
            nn.functional.cross_entropy(
                next_observation_logits.reshape(-1, next_observation_logits.size(-1)),
                next_observation_targets.reshape(-1),
            )
            + nn.functional.cross_entropy(reward_logits, reward_targets)
            + nn.functional.cross_entropy(terminated_logits, terminated_targets)
        )
        loss.backward()
        clip_gradients(model, config.max_grad_norm)
        optimizer.step()
        scheduler.step()

    train_metrics = _evaluate(model, dataset, device, batch_size=config.batch_size)
    held_out_metrics = (
        _evaluate(model, eval_dataset, device, batch_size=config.batch_size)
        if eval_dataset is not None
        else None
    )
    result = GridStepPredictionResult(
        train_case_count=len(dataset),
        eval_case_count=len(eval_dataset) if eval_dataset is not None else 0,
        initial_loss=initial_metrics.loss,
        final_loss=train_metrics.loss,
        final_next_observation_loss=train_metrics.next_observation_loss,
        final_reward_loss=train_metrics.reward_loss,
        final_terminated_loss=train_metrics.terminated_loss,
        per_cell_accuracy=train_metrics.observation.per_cell_accuracy,
        changed_cell_accuracy=train_metrics.observation.changed_cell_accuracy,
        whole_grid_match=train_metrics.observation.whole_grid_match,
        next_agent_cell_accuracy=train_metrics.observation.next_agent_cell_accuracy,
        reward_accuracy=train_metrics.reward_accuracy,
        terminated_accuracy=train_metrics.terminated_accuracy,
        eval_loss=held_out_metrics.loss if held_out_metrics is not None else None,
        eval_next_observation_loss=held_out_metrics.next_observation_loss if held_out_metrics is not None else None,
        eval_reward_loss=held_out_metrics.reward_loss if held_out_metrics is not None else None,
        eval_terminated_loss=held_out_metrics.terminated_loss if held_out_metrics is not None else None,
        eval_per_cell_accuracy=held_out_metrics.observation.per_cell_accuracy if held_out_metrics is not None else None,
        eval_changed_cell_accuracy=held_out_metrics.observation.changed_cell_accuracy if held_out_metrics is not None else None,
        eval_whole_grid_match=held_out_metrics.observation.whole_grid_match if held_out_metrics is not None else None,
        eval_next_agent_cell_accuracy=held_out_metrics.observation.next_agent_cell_accuracy if held_out_metrics is not None else None,
        eval_reward_accuracy=held_out_metrics.reward_accuracy if held_out_metrics is not None else None,
        eval_terminated_accuracy=held_out_metrics.terminated_accuracy if held_out_metrics is not None else None,
        max_steps=config.max_steps,
    )
    return GridStepTrainingArtifacts(
        result=result,
        model=model,
        config=config,
        grid_size=(dataset.height, dataset.width),
    )


def _evaluate(
    model: GridStepPredictionModel,
    dataset: GridStepPredictionDataset,
    device: torch.device,
    *,
    batch_size: int,
) -> _SplitMetrics:
    model.eval()
    predicted_class_ids = []
    agent_scores = []
    total_next_observation_loss = 0.0
    total_reward_loss = 0.0
    total_terminated_loss = 0.0
    total_reward_correct = 0
    total_terminated_correct = 0
    total_count = 0
    total_cell_count = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            observations = torch.stack([item[0] for item in batch]).to(device)
            action_ids = torch.stack([item[1] for item in batch]).to(device)
            next_observation_targets = torch.stack([item[2] for item in batch]).to(device)
            reward_targets = torch.stack([item[3] for item in batch]).to(device)
            terminated_targets = torch.stack([item[4] for item in batch]).to(device)
            next_observation_logits, reward_logits, terminated_logits = model(observations, action_ids)
            total_next_observation_loss += float(
                nn.functional.cross_entropy(
                    next_observation_logits.reshape(-1, next_observation_logits.size(-1)),
                    next_observation_targets.reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_reward_loss += float(
                nn.functional.cross_entropy(reward_logits, reward_targets, reduction="sum").item()
            )
            total_terminated_loss += float(
                nn.functional.cross_entropy(terminated_logits, terminated_targets, reduction="sum").item()
            )
            total_reward_correct += int((reward_logits.argmax(dim=1) == reward_targets).sum().item())
            total_terminated_correct += int((terminated_logits.argmax(dim=1) == terminated_targets).sum().item())
            total_count += int(reward_targets.numel())
            total_cell_count += int(next_observation_targets.numel())
            predicted_class_ids.append(next_observation_logits.argmax(dim=2).cpu())
            agent_scores.append(
                nn.functional.softmax(next_observation_logits, dim=2)[:, :, _AGENT_CLASS_ID].cpu()
            )
    model.train()
    observation_metrics = next_observation_metrics(
        torch.cat(predicted_class_ids),
        torch.cat(agent_scores),
        dataset.examples,
        width=dataset.width,
    )
    next_observation_loss = total_next_observation_loss / total_cell_count
    reward_loss = total_reward_loss / total_count
    terminated_loss = total_terminated_loss / total_count
    return _SplitMetrics(
        loss=next_observation_loss + reward_loss + terminated_loss,
        next_observation_loss=next_observation_loss,
        reward_loss=reward_loss,
        terminated_loss=terminated_loss,
        observation=observation_metrics,
        reward_accuracy=total_reward_correct / total_count,
        terminated_accuracy=total_terminated_correct / total_count,
    )
