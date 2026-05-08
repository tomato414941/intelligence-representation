from __future__ import annotations

import torch


def load_compatible_module_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    module_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Initialize selected modules from another model state.

    This is a transfer-initialization helper, not a checkpoint restore path.
    Full checkpoint loads should use strict state dict loading.
    """
    loaded_names: list[str] = []
    for module_name in module_names:
        module = getattr(model, module_name)
        module_state = module.state_dict()
        prefix = f"{module_name}."
        compatible_state: dict[str, torch.Tensor] = {}
        for name, value in state_dict.items():
            if not name.startswith(prefix):
                continue
            module_key = name.removeprefix(prefix)
            if module_key in module_state and module_state[module_key].shape == value.shape:
                compatible_state[module_key] = value
        module.load_state_dict(compatible_state, strict=False)
        loaded_names.extend(f"{module_name}.{name}" for name in compatible_state)
    return tuple(sorted(loaded_names))
