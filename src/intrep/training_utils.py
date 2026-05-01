from __future__ import annotations

import torch


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


def clip_gradients(model: torch.nn.Module, max_norm: float | None) -> float | None:
    if max_norm is None:
        return None
    if max_norm <= 0.0:
        raise ValueError("max_norm must be positive")
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())


def _no_decay_parameter_names(model: torch.nn.Module) -> set[str]:
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
            for parameter_name, _ in module.named_parameters(recurse=False):
                names.add(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    return names
