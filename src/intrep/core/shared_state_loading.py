from __future__ import annotations

import torch


def load_compatible_shared_state(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Initialize matching components from another model state.

    This is a transfer-initialization helper, not a checkpoint restore path.
    Full checkpoint loads should use strict state dict loading.
    """
    current_state = model.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in current_state and current_state[name].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)
    return tuple(sorted(compatible_state))
