from __future__ import annotations

import torch


def load_compatible_shared_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Load matching components and skip missing or shape-incompatible state."""
    current_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in current_state and current_state[name].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)
    return tuple(sorted(compatible_state))
