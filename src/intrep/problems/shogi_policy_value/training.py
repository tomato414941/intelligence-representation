from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Callable, Literal, Sequence
import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader

from intrep.core.training_run import BestMetricTracker
from intrep.core.training_utils import build_adamw
from intrep.problems.shogi_policy_value.examples import (
    LegalMoveTokenPolicyValueBatch,
    PolicyPlaneValueBatch,
    ShogiPolicyPlaneValueDataset,
    ShogiLegalMoveTokenPolicyValueDataset,
    ShogiPolicyValueDatasetItem,
    collate_legal_move_token_policy_value_samples,
    collate_policy_plane_value_samples,
)
from intrep.problems.shogi_policy_value.model import (
    PolicyPlaneShogiPolicyValueModel,
    PolicyPlaneShogiPolicyValueModelConfig,
    SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_POSITION_INPUT_MODULE_ID,
    SHOGI_SHARED_CORE_MODULE_ID,
    SHOGI_STATE_TOKEN_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    SHOGI_VALUE_OUTPUT_MODULE_ID,
    SharedCoreShogiPolicyValueModel,
    SharedCoreShogiPolicyValueModelConfig,
    validate_shogi_policy_value_components,
)
from intrep.representation.outputs.shogi_legal_move_token import ShogiStateTokenLegalMovePolicyOutput
from intrep.worlds.shogi.position_encoding import ShogiPositionFeatures


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
    input: str = SHOGI_POSITION_INPUT_MODULE_ID
    core: str = SHOGI_SHARED_CORE_MODULE_ID
    policy_output: str = SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID
    value_output: str = SHOGI_VALUE_OUTPUT_MODULE_ID
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    allow_nonstandard_loss_weights: bool = False
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


@dataclass(frozen=True)
class ShogiPolicyValuePhaseProgress:
    phase: str
    event: Literal["start", "progress", "done"]
    elapsed_seconds: float
    processed_batches: int
    total_batches: int
    processed_examples: int
    device: str


def train_shogi_policy_value_model(
    examples: Sequence[ShogiPolicyValueDatasetItem],
    *,
    eval_examples: Sequence[ShogiPolicyValueDatasetItem] | None = None,
    config: ShogiPolicyValueTrainingConfig | None = None,
    initial_state_dict: object | None = None,
    progress_callback: Callable[[ShogiPolicyValueTrainingProgress], None] | None = None,
    phase_progress_callback: Callable[[ShogiPolicyValuePhaseProgress], None] | None = None,
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
    validate_shogi_policy_value_components(
        input=training_config.input,
        core=training_config.core,
        policy_output=training_config.policy_output,
        value_output=training_config.value_output,
    )
    validate_shogi_policy_value_loss_weights(training_config)
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)
    dataset = _build_shogi_policy_value_dataset(examples, training_config)
    loader = _build_shogi_policy_value_loader(dataset, training_config, shuffle=True)
    train_eval_examples = _limit_examples(examples, training_config.max_train_eval_examples)
    train_eval_dataset = _build_shogi_policy_value_dataset(train_eval_examples, training_config)
    train_eval_loader = _build_shogi_policy_value_loader(train_eval_dataset, training_config, shuffle=False)
    limited_eval_examples = (
        _limit_examples(eval_examples, training_config.max_eval_examples) if eval_examples is not None else None
    )
    eval_dataset = (
        _build_shogi_policy_value_dataset(limited_eval_examples, training_config)
        if limited_eval_examples is not None
        else None
    )
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
        phase="initial_train_eval",
        progress_every_batches=training_config.log_every,
        progress_callback=phase_progress_callback,
    )
    initial_eval_metrics: ShogiPolicyValueEvaluationMetrics | None = None
    best_eval_tracker = BestMetricTracker(mode="min")
    best_model_state_dict: dict[str, torch.Tensor] | None = None
    if eval_loader is not None:
        initial_eval_metrics = evaluate_shogi_policy_value_metrics(
            model,
            eval_loader,
            phase="initial_eval",
            progress_every_batches=training_config.log_every,
            progress_callback=phase_progress_callback,
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
        for batch in loader:
            batch_ready = time.monotonic()
            interval_data_wait_seconds += batch_ready - last_batch_finished
            batch = _batch_to_device(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            value_mask = torch.isfinite(batch.value_targets)
            forward_backward_started = time.monotonic()
            if training_config.policy_loss_weight == 0.0 and training_config.value_loss_weight > 0.0 and value_mask.any():
                value_predictions = model.predict_value(batch.position_features)
                loss = torch.zeros((), dtype=value_predictions.dtype, device=device)
            elif training_config.value_loss_weight > 0.0 and value_mask.any():
                logits, value_predictions = _forward_batch_policy_value(model, batch)
                policy_loss = _batch_policy_target_loss(logits, batch)
                loss = training_config.policy_loss_weight * policy_loss
            else:
                logits = _forward_batch_policy(model, batch)
                value_predictions = None
                policy_loss = _batch_policy_target_loss(logits, batch)
                loss = training_config.policy_loss_weight * policy_loss
            if value_predictions is not None:
                value_loss = torch.nn.functional.mse_loss(
                    value_predictions[value_mask],
                    batch.value_targets[value_mask],
                )
                loss = loss + training_config.value_loss_weight * value_loss
            loss.backward()
            forward_backward_finished = time.monotonic()
            interval_forward_backward_seconds += forward_backward_finished - forward_backward_started
            optimizer.step()
            optimizer_finished = time.monotonic()
            interval_optimizer_seconds += optimizer_finished - forward_backward_finished
            step += 1
            eval_step_metrics: ShogiPolicyValueEvaluationMetrics | None = None
            if training_config.eval_every is not None and step % training_config.eval_every == 0:
                eval_step_metrics = evaluate_shogi_policy_value_metrics(
                    model,
                    eval_loader,
                    phase=f"step_{step}_eval",
                    progress_every_batches=training_config.log_every,
                    progress_callback=phase_progress_callback,
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
        phase="final_train_eval",
        progress_every_batches=training_config.log_every,
        progress_callback=phase_progress_callback,
    )
    eval_metrics: ShogiPolicyValueEvaluationMetrics | None = None
    if eval_loader is not None:
        eval_metrics = evaluate_shogi_policy_value_metrics(
            model,
            eval_loader,
            phase="final_eval",
            progress_every_batches=training_config.log_every,
            progress_callback=phase_progress_callback,
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
    loader: DataLoader,
) -> tuple[float, float]:
    metrics = evaluate_shogi_policy_value_metrics(model, loader)
    return metrics.loss, metrics.accuracy


def validate_shogi_policy_value_loss_weights(config: ShogiPolicyValueTrainingConfig) -> None:
    if config.policy_loss_weight < 0.0:
        raise ValueError("policy_loss_weight must be non-negative")
    if config.value_loss_weight < 0.0:
        raise ValueError("value_loss_weight must be non-negative")
    if config.policy_loss_weight == 0.0 and config.value_loss_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    if config.allow_nonstandard_loss_weights:
        warnings.warn(
            "NONSTANDARD shogi policy/value loss weights are enabled. "
            "This can create checkpoints that are not suitable for normal MCTS use. "
            "Use policy_loss_weight=1.0 and value_loss_weight=1.0 unless explicitly approved.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if config.policy_loss_weight != 1.0 or config.value_loss_weight != 1.0:
        raise ValueError(
            "policy_loss_weight and value_loss_weight default to 1.0; "
            "set allow_nonstandard_loss_weights=True to use other values"
        )


def evaluate_shogi_policy_value_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    phase: str | None = None,
    progress_every_batches: int | None = None,
    progress_callback: Callable[[ShogiPolicyValuePhaseProgress], None] | None = None,
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
    if phase is not None and progress_callback is not None:
        progress_callback(
            ShogiPolicyValuePhaseProgress(
                phase=phase,
                event="start",
                elapsed_seconds=0.0,
                processed_batches=0,
                total_batches=batch_count,
                processed_examples=0,
                device=str(device),
            )
        )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = _batch_to_device(batch, device=device)
            value_mask = torch.isfinite(batch.value_targets)
            if value_mask.any() and hasattr(model, "predict_value"):
                logits, value_predictions = _forward_batch_policy_value(model, batch)
            else:
                logits = _forward_batch_policy(model, batch)
                value_predictions = None
            loss = _batch_policy_target_loss(logits, batch)
            losses.append(float(loss.item()))
            predictions = logits.argmax(dim=1)
            correct += int((predictions == batch.labels).sum().item())
            sorted_indices = logits.argsort(dim=1, descending=True)
            label_matches = sorted_indices.eq(batch.labels[:, None])
            ranks = label_matches.float().argmax(dim=1) + 1
            top_3_correct += int((ranks <= 3).sum().item())
            top_5_correct += int((ranks <= 5).sum().item())
            reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
            rank_sum += float(ranks.float().sum().item())
            total += int(batch.labels.numel())
            if value_predictions is not None:
                value_loss = torch.nn.functional.mse_loss(
                    value_predictions[value_mask],
                    batch.value_targets[value_mask],
                )
                value_losses.append(float(value_loss.item()))
            if (
                phase is not None
                and progress_callback is not None
                and progress_every_batches is not None
                and batch_index % progress_every_batches == 0
            ):
                elapsed = time.monotonic() - started
                progress_callback(
                    ShogiPolicyValuePhaseProgress(
                        phase=phase,
                        event="progress",
                        elapsed_seconds=elapsed,
                        processed_batches=batch_index,
                        total_batches=batch_count,
                        processed_examples=total,
                        device=str(device),
                    )
                )
    if phase is not None and progress_callback is not None:
        elapsed = time.monotonic() - started
        progress_callback(
            ShogiPolicyValuePhaseProgress(
                phase=phase,
                event="done",
                elapsed_seconds=elapsed,
                processed_batches=batch_count,
                total_batches=batch_count,
                processed_examples=total,
                device=str(device),
            )
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
    dataset: ShogiLegalMoveTokenPolicyValueDataset | ShogiPolicyPlaneValueDataset,
    config: ShogiPolicyValueTrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    effective_shuffle = shuffle and not bool(getattr(dataset.examples, "sequential_access_preferred", False))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=effective_shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=_collate_shogi_policy_value_batch(dataset),
    )


def _build_shogi_policy_value_dataset(
    examples: Sequence[ShogiPolicyValueDatasetItem],
    config: ShogiPolicyValueTrainingConfig,
) -> ShogiLegalMoveTokenPolicyValueDataset | ShogiPolicyPlaneValueDataset:
    if config.policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return ShogiPolicyPlaneValueDataset(examples)
    return ShogiLegalMoveTokenPolicyValueDataset(examples)


ShogiPolicyValueBatch = LegalMoveTokenPolicyValueBatch | PolicyPlaneValueBatch


def _collate_shogi_policy_value_batch(
    dataset: ShogiLegalMoveTokenPolicyValueDataset | ShogiPolicyPlaneValueDataset,
):
    if isinstance(dataset, ShogiPolicyPlaneValueDataset):
        return collate_policy_plane_value_samples
    return collate_legal_move_token_policy_value_samples


def _batch_to_device(
    batch: ShogiPolicyValueBatch,
    *,
    device: torch.device,
) -> ShogiPolicyValueBatch:
    if isinstance(batch, (LegalMoveTokenPolicyValueBatch, PolicyPlaneValueBatch)):
        return batch.to(device)
    raise TypeError(f"unsupported shogi policy/value batch: {type(batch).__name__}")


def _policy_target_loss(logits: torch.Tensor, policy_targets: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    return -(policy_targets * log_probs).sum(dim=1).mean()


def _sparse_policy_target_loss(
    logits: torch.Tensor,
    target_action_indices: torch.Tensor,
    target_weights: torch.Tensor,
) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(dim=1, index=target_action_indices.long())
    return -(target_weights * target_log_probs).sum(dim=1).mean()


def _batch_policy_target_loss(logits: torch.Tensor, batch: ShogiPolicyValueBatch) -> torch.Tensor:
    if isinstance(batch, LegalMoveTokenPolicyValueBatch):
        return _policy_target_loss(logits, batch.policy_targets)
    if isinstance(batch, PolicyPlaneValueBatch):
        return _sparse_policy_target_loss(logits, batch.target_action_indices, batch.target_weights)
    raise TypeError(f"unsupported shogi policy/value batch: {type(batch).__name__}")


def _forward_batch_policy(model: nn.Module, batch: ShogiPolicyValueBatch) -> torch.Tensor:
    if isinstance(batch, LegalMoveTokenPolicyValueBatch):
        return model(batch.position_features, batch.legal_move_token_features, batch.legal_move_token_mask)
    if isinstance(batch, PolicyPlaneValueBatch):
        return model(batch.position_features, batch.legal_action_mask)
    raise TypeError(f"unsupported shogi policy/value batch: {type(batch).__name__}")


def _forward_batch_policy_value(model: nn.Module, batch: ShogiPolicyValueBatch) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, LegalMoveTokenPolicyValueBatch):
        return _forward_policy_value(
            model,
            batch.position_features,
            batch.legal_move_token_features,
            batch.legal_move_token_mask,
        )
    if isinstance(batch, PolicyPlaneValueBatch):
        return _forward_policy_value(model, batch.position_features, batch.legal_action_mask)
    raise TypeError(f"unsupported shogi policy/value batch: {type(batch).__name__}")


def _forward_policy_value(
    model: nn.Module,
    position_features: ShogiPositionFeatures,
    *policy_args: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "forward_policy_value"):
        return model.forward_policy_value(position_features, *policy_args)
    logits = model(position_features, *policy_args)
    value_predictions = model.predict_value(position_features)
    return logits, value_predictions


def _limit_examples(
    examples: Sequence[ShogiPolicyValueDatasetItem],
    max_examples: int | None,
) -> Sequence[ShogiPolicyValueDatasetItem]:
    if max_examples is None:
        return examples
    if max_examples <= 0:
        raise ValueError("max eval examples must be positive")
    return examples[:max_examples]


def build_shogi_policy_value_model(config: ShogiPolicyValueTrainingConfig) -> nn.Module:
    validate_shogi_policy_value_components(
        input=config.input,
        core=config.core,
        policy_output=config.policy_output,
        value_output=config.value_output,
    )
    if config.policy_output in (
        SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID,
        SHOGI_STATE_TOKEN_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    ):
        shared_config = SharedCoreShogiPolicyValueModelConfig(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
        policy_output = None
        if config.policy_output == SHOGI_STATE_TOKEN_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID:
            policy_output = ShogiStateTokenLegalMovePolicyOutput(
                embedding_dim=config.embedding_dim,
                hidden_dim=config.hidden_dim,
            )
        return SharedCoreShogiPolicyValueModel(
            shared_config,
            policy_output=policy_output,
        )
    if config.policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
            )
        )
    raise ValueError(f"unsupported shogi policy/value policy output: {config.policy_output}")
