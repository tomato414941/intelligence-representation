from __future__ import annotations

import math
from typing import Literal

import torch

LearningRateSchedule = Literal["constant", "warmup_cosine"]


def build_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    no_decay_names = _no_decay_parameter_names(model)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in no_decay_names or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: LearningRateSchedule,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    if schedule == "warmup_cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _warmup_cosine_lr_factor(
                step,
                warmup_steps=warmup_steps,
                max_steps=max_steps,
            ),
        )
    raise ValueError("lr_schedule must be one of: constant, warmup_cosine")


def clip_gradients(model: torch.nn.Module, max_norm: float | None) -> float | None:
    if max_norm is None:
        return None
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())


def _warmup_cosine_lr_factor(step: int, *, warmup_steps: int, max_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    decay_steps = max(1, max_steps - warmup_steps)
    decay_step = min(max(0, step - warmup_steps), decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))


def _no_decay_parameter_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            for parameter_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names
