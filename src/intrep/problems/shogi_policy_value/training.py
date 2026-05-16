from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Callable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from intrep.core.training_run import BestMetricTracker
from intrep.core.training_utils import build_adamw
from intrep.problems.shogi_policy_value.examples import (
    ShogiPolicyValueDataset,
    ShogiPolicyValueDatasetItem,
    ShogiPolicyValueExample,
)
from intrep.problems.shogi_policy_value.model import (
    SharedCoreShogiPolicyValueModel,
    SharedCoreShogiPolicyValueModelConfig,
    ShogiPolicyValueModel,
    ShogiPolicyValueModelConfig,
)


@dataclass(frozen=True)
class ShogiPolicyValueTrainingConfig:
    max_steps: int = 100
    batch_size: int = 8
    learning_rate: float = 0.003
    weight_decay: float = 0.0
    seed: int = 7
    embedding_dim: int = 256
    hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 6
    use_shared_core: bool = True
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.0
    device: str = "cpu"
    max_train_eval_examples: int | None = None
    max_eval_examples: int | None = None
    log_every: int | None = None
    num_workers: int = 0
    pin_memory: bool = False
    progress_every: int | None = None
    eval_every: int | None = None
    early_stopping_patience: int | None = None


@dataclass(frozen=True)
class ShogiPolicyValueEvaluationMetrics:
    loss: float
    accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    mean_reciprocal_rank: float
    mean_correct_move_rank: float
    value_loss: float | None = None


@dataclass(frozen=True)
class ShogiPolicyValueTrainingMetrics:
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
    best_eval_loss: float | None
    best_eval_step: int | None
    max_steps: int
    actual_steps: int
    stopped_early: bool
    stopped_step: int | None
    early_stopping_patience: int | None


@dataclass(frozen=True)
class ShogiPolicyValueTrainingResult:
    model: nn.Module
    config: ShogiPolicyValueTrainingConfig
    metrics: ShogiPolicyValueTrainingMetrics
    best_model_state_dict: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class ShogiPolicyValueTrainingProgress:
    step: int
    max_steps: int
    loss: float
    elapsed_seconds: float
    data_wait_seconds: float
    forward_backward_seconds: float
    optimizer_seconds: float
    model: nn.Module
    config: ShogiPolicyValueTrainingConfig
    eval_metrics: ShogiPolicyValueEvaluationMetrics | None = None


def train_shogi_policy_value_model(
    examples: Sequence[ShogiPolicyValueDatasetItem],
    *,
    eval_examples: Sequence[ShogiPolicyValueDatasetItem] | None = None,
    config: ShogiPolicyValueTrainingConfig | None = None,
    initial_state_dict: object | None = None,
    progress_callback: Callable[[ShogiPolicyValueTrainingProgress], None] | None = None,
) -> ShogiPolicyValueTrainingResult:
    training_config = config or ShogiPolicyValueTrainingConfig()
    if training_config.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if training_config.log_every is not None and training_config.log_every <= 0:
        raise ValueError("log_every must be positive")
    if training_config.progress_every is not None and training_config.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if training_config.eval_every is not None and training_config.eval_every <= 0:
        raise ValueError("eval_every must be positive")
    if training_config.early_stopping_patience is not None and training_config.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive")
    if training_config.early_stopping_patience is not None and training_config.eval_every is None:
        raise ValueError("eval_every is required when early_stopping_patience is set")
    if training_config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if training_config.policy_loss_weight < 0.0:
        raise ValueError("policy_loss_weight must be non-negative")
    if training_config.value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be non-negative")
    if training_config.policy_loss_weight == 0.0 and training_config.value_loss_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)
    dataset = ShogiPolicyValueDataset(examples)
    loader = _build_shogi_policy_value_loader(dataset, training_config, shuffle=True)
    train_eval_examples = _limit_examples(examples, training_config.max_train_eval_examples)
    train_eval_dataset = ShogiPolicyValueDataset(train_eval_examples)
    train_eval_loader = _build_shogi_policy_value_loader(train_eval_dataset, training_config, shuffle=False)
    limited_eval_examples = (
        _limit_examples(eval_examples, training_config.max_eval_examples) if eval_examples is not None else None
    )
    eval_dataset = ShogiPolicyValueDataset(limited_eval_examples) if limited_eval_examples is not None else None
    eval_loader = (
        _build_shogi_policy_value_loader(eval_dataset, training_config, shuffle=False)
        if eval_dataset is not None
        else None
    )
    if training_config.eval_every is not None and eval_loader is None:
        raise ValueError("eval examples are required when eval_every is set")
    model = build_shogi_policy_value_model(training_config).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    optimizer = build_adamw(
        model,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    initial_metrics = evaluate_shogi_policy_value_metrics(
        model,
        train_eval_loader,
        log_label="initial_train_eval" if training_config.log_every is not None else None,
        log_every_batches=training_config.log_every,
    )
    initial_eval_metrics: ShogiPolicyValueEvaluationMetrics | None = None
    best_eval_tracker = BestMetricTracker(mode="min")
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    if eval_loader is not None:
        initial_eval_metrics = evaluate_shogi_policy_value_metrics(
            model,
            eval_loader,
            log_label="initial_eval" if training_config.log_every is not None else None,
            log_every_batches=training_config.log_every,
        )
        best_eval_tracker.update(step=0, value=initial_eval_metrics.loss)
        best_model_state_dict = copy.deepcopy(model.state_dict())

    model.train()
    step = 0
    no_improvement_eval_count = 0
    stopped_early = False
    started = time.monotonic()
    last_batch_finished = time.monotonic()
    interval_data_wait_seconds = 0.0
    interval_forward_backward_seconds = 0.0
    interval_optimizer_seconds = 0.0
    while step < training_config.max_steps:
        for position_token_ids, candidate_move_features, candidate_mask, labels, policy_targets, value_targets in loader:
            batch_ready = time.monotonic()
            interval_data_wait_seconds += batch_ready - last_batch_finished
            position_token_ids = position_token_ids.to(device)
            candidate_move_features = candidate_move_features.to(device)
            candidate_mask = candidate_mask.to(device)
            labels = labels.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            value_mask = torch.isfinite(value_targets)
            forward_backward_started = time.monotonic()
            if training_config.policy_loss_weight == 0.0 and training_config.value_loss_weight > 0.0 and value_mask.any():
                value_predictions = model.predict_value(position_token_ids)
                loss = torch.zeros((), dtype=value_predictions.dtype, device=device)
            elif training_config.value_loss_weight > 0.0 and value_mask.any():
                logits, value_predictions = _forward_policy_value(
                    model,
                    position_token_ids,
                    candidate_move_features,
                    candidate_mask,
                )
                policy_loss = _policy_target_loss(logits, policy_targets)
                loss = training_config.policy_loss_weight * policy_loss
            else:
                logits = model(position_token_ids, candidate_move_features, candidate_mask)
                value_predictions = None
                policy_loss = _policy_target_loss(logits, policy_targets)
                loss = training_config.policy_loss_weight * policy_loss
            if value_predictions is not None:
                value_loss = torch.nn.functional.mse_loss(value_predictions[value_mask], value_targets[value_mask])
                loss = loss + training_config.value_loss_weight * value_loss
            loss.backward()
            forward_backward_finished = time.monotonic()
            interval_forward_backward_seconds += forward_backward_finished - forward_backward_started
            optimizer.step()
            optimizer_finished = time.monotonic()
            interval_optimizer_seconds += optimizer_finished - forward_backward_finished
            step += 1
            if training_config.log_every is not None and step % training_config.log_every == 0:
                _log_training_progress(
                    step,
                    training_config.max_steps,
                    started,
                    loss,
                    device,
                    data_wait_seconds=interval_data_wait_seconds,
                    forward_backward_seconds=interval_forward_backward_seconds,
                    optimizer_seconds=interval_optimizer_seconds,
                )
            eval_step_metrics: ShogiPolicyValueEvaluationMetrics | None = None
            if training_config.eval_every is not None and step % training_config.eval_every == 0:
                eval_step_metrics = evaluate_shogi_policy_value_metrics(
                    model,
                    eval_loader,
                    log_label=f"step_{step}_eval" if training_config.log_every is not None else None,
                    log_every_batches=training_config.log_every,
                )
                if best_eval_tracker.update(step=step, value=eval_step_metrics.loss):
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    no_improvement_eval_count = 0
                else:
                    no_improvement_eval_count += 1
                model.train()
            if (
                progress_callback is not None
                and training_config.progress_every is not None
                and step % training_config.progress_every == 0
            ):
                progress_callback(
                    ShogiPolicyValueTrainingProgress(
                        step=step,
                        max_steps=training_config.max_steps,
                        loss=float(loss.detach().cpu().item()),
                        elapsed_seconds=time.monotonic() - started,
                        data_wait_seconds=interval_data_wait_seconds,
                        forward_backward_seconds=interval_forward_backward_seconds,
                        optimizer_seconds=interval_optimizer_seconds,
                        model=model,
                        config=training_config,
                        eval_metrics=eval_step_metrics,
                    )
                )
                interval_data_wait_seconds = 0.0
                interval_forward_backward_seconds = 0.0
                interval_optimizer_seconds = 0.0
            last_batch_finished = time.monotonic()
            if (
                training_config.early_stopping_patience is not None
                and no_improvement_eval_count >= training_config.early_stopping_patience
            ):
                stopped_early = True
                break
            if step >= training_config.max_steps:
                break
        if stopped_early:
            break

    final_metrics = evaluate_shogi_policy_value_metrics(
        model,
        train_eval_loader,
        log_label="final_train_eval" if training_config.log_every is not None else None,
        log_every_batches=training_config.log_every,
    )
    eval_metrics: ShogiPolicyValueEvaluationMetrics | None = None
    if eval_loader is not None:
        eval_metrics = evaluate_shogi_policy_value_metrics(
            model,
            eval_loader,
            log_label="final_eval" if training_config.log_every is not None else None,
            log_every_batches=training_config.log_every,
        )
        if best_eval_tracker.update(step=step, value=eval_metrics.loss):
            best_model_state_dict = copy.deepcopy(model.state_dict())
    return ShogiPolicyValueTrainingResult(
        model=model,
        config=training_config,
        metrics=ShogiPolicyValueTrainingMetrics(
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
            best_eval_loss=best_eval_tracker.best.value if best_eval_tracker.best is not None else None,
            best_eval_step=best_eval_tracker.best.step if best_eval_tracker.best is not None else None,
            max_steps=training_config.max_steps,
            actual_steps=step,
            stopped_early=stopped_early,
            stopped_step=step if stopped_early else None,
            early_stopping_patience=training_config.early_stopping_patience,
        ),
        best_model_state_dict=best_model_state_dict,
    )


def evaluate_shogi_policy_value_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    metrics = evaluate_shogi_policy_value_metrics(model, loader)
    return metrics.loss, metrics.accuracy


def evaluate_shogi_policy_value_metrics(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    log_label: str | None = None,
    log_every_batches: int | None = None,
) -> ShogiPolicyValueEvaluationMetrics:
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
    started = time.monotonic()
    batch_count = len(loader)
    if log_label is not None:
        print(f"{log_label} start batches={batch_count} device={device}", flush=True)
    with torch.no_grad():
        for batch_index, (
            position_token_ids,
            candidate_move_features,
            candidate_mask,
            labels,
            policy_targets,
            value_targets,
        ) in enumerate(loader, start=1):
            position_token_ids = position_token_ids.to(device)
            candidate_move_features = candidate_move_features.to(device)
            candidate_mask = candidate_mask.to(device)
            labels = labels.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            value_mask = torch.isfinite(value_targets)
            if value_mask.any() and hasattr(model, "predict_value"):
                logits, value_predictions = _forward_policy_value(
                    model,
                    position_token_ids,
                    candidate_move_features,
                    candidate_mask,
                )
            else:
                logits = model(position_token_ids, candidate_move_features, candidate_mask)
                value_predictions = None
            loss = _policy_target_loss(logits, policy_targets)
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
            if value_predictions is not None:
                value_loss = torch.nn.functional.mse_loss(value_predictions[value_mask], value_targets[value_mask])
                value_losses.append(float(value_loss.item()))
            if log_label is not None and log_every_batches is not None and batch_index % log_every_batches == 0:
                elapsed = time.monotonic() - started
                batches_per_second = batch_index / elapsed if elapsed > 0.0 else 0.0
                print(
                    f"{log_label} batch={batch_index}/{batch_count}"
                    f" examples={total}"
                    f" elapsed_seconds={elapsed:.1f}"
                    f" batches_per_second={batches_per_second:.3f}",
                    flush=True,
                )
    if log_label is not None:
        elapsed = time.monotonic() - started
        batches_per_second = batch_count / elapsed if elapsed > 0.0 else 0.0
        print(
            f"{log_label} done batches={batch_count}"
            f" examples={total}"
            f" elapsed_seconds={elapsed:.1f}"
            f" batches_per_second={batches_per_second:.3f}",
            flush=True,
        )
    return ShogiPolicyValueEvaluationMetrics(
        loss=sum(losses) / len(losses),
        accuracy=correct / total,
        top_3_accuracy=top_3_correct / total,
        top_5_accuracy=top_5_correct / total,
        mean_reciprocal_rank=reciprocal_rank_sum / total,
        mean_correct_move_rank=rank_sum / total,
        value_loss=sum(value_losses) / len(value_losses) if value_losses else None,
    )


def _build_shogi_policy_value_loader(
    dataset: ShogiPolicyValueDataset,
    config: ShogiPolicyValueTrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    effective_shuffle = shuffle and not bool(getattr(dataset.examples, "sequential_access_preferred", False))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=effective_shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )


def _policy_target_loss(logits: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(policy_targets * log_probs).sum(dim=1).mean()


def _forward_policy_value(
    model: nn.Module,
    position_token_ids: torch.Tensor,
    candidate_move_features: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_policy_value"):
        return model.forward_policy_value(position_token_ids, candidate_move_features, candidate_mask)
    logits = model(position_token_ids, candidate_move_features, candidate_mask)
    value_predictions = model.predict_value(position_token_ids)
    return logits, value_predictions


def _limit_examples(
    examples: Sequence[ShogiPolicyValueExample],
    max_examples: int | None,
) -> Sequence[ShogiPolicyValueExample]:
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
    *,
    data_wait_seconds: float,
    forward_backward_seconds: float,
    optimizer_seconds: float,
) -> None:
    elapsed = time.monotonic() - started
    steps_per_second = step / elapsed if elapsed > 0.0 else 0.0
    parts = [
        f"step={step}/{max_steps}",
        f"elapsed_seconds={elapsed:.1f}",
        f"steps_per_second={steps_per_second:.3f}",
        f"loss={float(loss.detach().item()):.4f}",
        f"data_wait_seconds={data_wait_seconds:.3f}",
        f"forward_backward_seconds={forward_backward_seconds:.3f}",
        f"optimizer_seconds={optimizer_seconds:.3f}",
        f"device={device}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        parts.append(f"cuda_max_memory_mb={allocated_mb:.1f}")
    print(" ".join(parts), flush=True)


def build_shogi_policy_value_model(config: ShogiPolicyValueTrainingConfig) -> nn.Module:
    if config.use_shared_core:
        return SharedCoreShogiPolicyValueModel(
            SharedCoreShogiPolicyValueModelConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        )
    return ShogiPolicyValueModel(
        ShogiPolicyValueModelConfig(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
        )
    )
