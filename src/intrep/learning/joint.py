from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from torch import nn


class JointTrainer:
    """Accumulate every registered source, then update the entire shared model once."""

    def __init__(self, model: nn.Module, weights: Mapping[str, float], *, learning_rate: float,
                 optimizer: str = "adamw", weight_decay: float = 0.0, momentum: float = 0.0,
                 max_grad_norm: float = 1.0) -> None:
        if (not weights or any(not name or name in {"weighted_loss", "grad_norm"}
                               or not math.isfinite(weight) or weight <= 0
                               for name, weight in weights.items())
                or not math.isfinite(sum(weights.values()))
                or not math.isfinite(learning_rate) or learning_rate <= 0
                or not math.isfinite(weight_decay) or weight_decay < 0
                or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
                or not math.isfinite(momentum) or not 0 <= momentum < 1):
            raise ValueError("invalid joint training configuration")
        self.model = model
        self.weights = dict(weights)
        self.max_grad_norm = max_grad_norm
        self.steps = 0
        parameters = list(model.parameters())
        if optimizer == "adamw":
            if momentum:
                raise ValueError("momentum is only configured for SGD")
            self.optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay, foreach=False)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum,
                                             weight_decay=weight_decay, foreach=False)
        else:
            raise ValueError("choose adamw or sgd")
        self.synchronize_parameters()

    def synchronize_parameters(self) -> None:
        """Keep core optimizer state; replace only state belonging to detached weights."""
        parameters = list(self.model.parameters())
        if not parameters or any(not parameter.requires_grad for parameter in parameters):
            raise ValueError("joint learning requires every attached parameter to remain trainable")
        active = {id(parameter) for parameter in parameters}
        for parameter in list(self.optimizer.state):
            if id(parameter) not in active:
                del self.optimizer.state[parameter]
        self.optimizer.param_groups[0]["params"] = parameters

    def add_source(self, name: str, weight: float = 1.0) -> None:
        if (not name or name in self.weights or name in {"weighted_loss", "grad_norm"}
                or not math.isfinite(weight) or weight <= 0 or not math.isfinite(sum(self.weights.values()) + weight)):
            raise ValueError("a new source requires a unique name and positive weight")
        self.weights[name] = weight

    def step(self, losses: Mapping[str, Callable[[], torch.Tensor]]) -> dict[str, float]:
        if set(losses) != set(self.weights):
            raise ValueError("each joint update must include every registered data source")
        self.synchronize_parameters()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_weight = sum(self.weights.values())
        metrics = {}
        device_groups = {}
        try:
            # Release each source's activations after backward; gradients accumulate.
            for name, weight in self.weights.items():
                loss = losses[name]()
                if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                        or not loss.requires_grad):
                    raise ValueError(f"source {name!r} must produce a finite differentiable scalar loss")
                metrics[name] = loss.detach()
                device_groups.setdefault(loss.device, []).append(name)
                (loss * (weight / total_weight)).backward()
            # Transfer loss scalars together instead of synchronizing CUDA for
            # every source. Reject invalid losses before any parameter update.
            for names in device_groups.values():
                values = torch.stack([metrics[name] for name in names]).cpu().tolist()
                for name, value in zip(names, values):
                    if not math.isfinite(value):
                        raise ValueError(f"source {name!r} must produce a finite differentiable scalar loss")
                    metrics[name] = value
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm,
                                            error_if_nonfinite=True, foreach=False)
            self.optimizer.step()
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.steps += 1
        metrics["weighted_loss"] = sum(metrics[name] * weight for name, weight in self.weights.items()) / total_weight
        metrics["grad_norm"] = float(norm)
        return metrics

    def state_dict(self) -> dict:
        self.synchronize_parameters()
        return {"steps": self.steps, "weights": self.weights.copy(),
                "max_grad_norm": self.max_grad_norm,
                "optimizer_type": type(self.optimizer).__name__,
                "parameter_names": [name for name, _ in self.model.named_parameters()],
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.synchronize_parameters()
        if (state["weights"] != self.weights or state["max_grad_norm"] != self.max_grad_norm
                or state["optimizer_type"] != type(self.optimizer).__name__
                or state["parameter_names"] != [name for name, _ in self.model.named_parameters()]):
            raise ValueError("joint checkpoint sources, optimizer or module identities differ")
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = state["steps"]
