from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class SharedPredictor(nn.Module):
    """Exchangeable input/output modules around one persistent learned core.

    Raw input formats and losses belong to callers. Only embedding sequences
    and hidden states cross the core boundary; head names are not modalities.
    """

    def __init__(self, core: nn.Module, dimension: int) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("the core dimension must be positive")
        self.core = core
        self.dimension = dimension
        self.input_heads = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()

    def _attach(self, heads: nn.ModuleDict, name: str, head: nn.Module) -> None:
        if not name or "." in name or not isinstance(head, nn.Module):
            raise ValueError("heads require a nonempty module name without dots")
        reference = next(self.core.parameters())
        head.to(device=reference.device, dtype=reference.dtype)
        head.train(self.training)
        heads[name] = head

    def attach_input(self, name: str, head: nn.Module) -> None:
        self._attach(self.input_heads, name, head)

    def attach_output(self, name: str, head: nn.Module) -> None:
        self._attach(self.output_heads, name, head)

    def detach_input(self, name: str) -> nn.Module:
        return self.input_heads.pop(name)

    def detach_output(self, name: str) -> nn.Module:
        return self.output_heads.pop(name)

    def encode(self, name: str, *args, **kwargs) -> torch.Tensor:
        embeddings = self.input_heads[name](*args, **kwargs)
        self._sequence(embeddings)
        return embeddings

    def _sequence(self, sequence: torch.Tensor) -> None:
        if (not isinstance(sequence, torch.Tensor) or sequence.ndim != 3
                or sequence.shape[-1] != self.dimension or min(sequence.shape[:2]) < 1):
            raise ValueError("the core boundary requires [batch, sequence, hidden] embeddings")

    def forward(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        self._sequence(embeddings)
        return self.core(embeddings, **kwargs)

    def decode(self, name: str, hidden: torch.Tensor, *args, **kwargs):
        self._sequence(hidden)
        return self.output_heads[name](hidden, *args, **kwargs)

    def parameter_aliases(self) -> list[list[str]]:
        aliases: dict[int, list[str]] = {}
        for name, parameter in self.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        return sorted(sorted(names) for names in aliases.values() if len(names) > 1)

    def module_state_dict(self) -> dict:
        """Store modules separately; constructors remain explicit in the caller."""
        return {
            "dimension": self.dimension,
            "core": self.core.state_dict(),
            "inputs": {name: head.state_dict() for name, head in self.input_heads.items()},
            "outputs": {name: head.state_dict() for name, head in self.output_heads.items()},
            "parameter_aliases": self.parameter_aliases(),
        }

    def load_module_state_dict(self, state: Mapping) -> None:
        if (state["dimension"] != self.dimension
                or set(state["inputs"]) != set(self.input_heads)
                or set(state["outputs"]) != set(self.output_heads)
                or state["parameter_aliases"] != self.parameter_aliases()):
            raise ValueError("checkpoint module identities or tied parameters differ")
        modules = [(self.core, state["core"])]
        modules.extend((head, state["inputs"][name]) for name, head in self.input_heads.items())
        modules.extend((head, state["outputs"][name]) for name, head in self.output_heads.items())
        # Validate every module before modifying any parameters.
        for module, saved in modules:
            current = module.state_dict()
            if (set(current) != set(saved)
                    or any(current[key].shape != saved[key].shape for key in current)):
                raise ValueError("checkpoint module parameter shapes differ")
        flat = {f"core.{key}": value for key, value in state["core"].items()}
        for group, prefix in (("inputs", "input_heads"), ("outputs", "output_heads")):
            for name, values in state[group].items():
                flat.update({f"{prefix}.{name}.{key}": value for key, value in values.items()})
        for aliases in self.parameter_aliases():
            if any(not torch.equal(flat[aliases[0]], flat[name]) for name in aliases[1:]):
                raise ValueError("checkpoint contains conflicting values for tied parameters")
        for module, saved in modules:
            module.load_state_dict(saved, strict=True)
