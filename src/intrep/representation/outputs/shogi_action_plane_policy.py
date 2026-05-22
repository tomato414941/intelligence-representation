from __future__ import annotations

import torch
from torch import nn

from intrep.representation.outputs.shogi_action_plane_policy_encoding import SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT
from intrep.representation.shogi_position_hidden import ShogiPositionHiddenLayout


class ShogiActionPlanePolicyHead(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        action_count: int,
        position_layout: ShogiPositionHiddenLayout,
    ) -> None:
        super().__init__()
        if action_count % SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT != 0:
            raise ValueError("action_count must be divisible by the shogi action-plane move type count")
        self.position_layout = position_layout
        self.square_count = action_count // SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT
        if self.square_count != position_layout.square_element_count:
            raise ValueError("action_count square count must match the position layout square count")
        self.state_action_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_count),
        )
        self.square_move_type_scorer = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT),
        )
        self.action_logit_bias = nn.Parameter(torch.zeros(action_count))

    def forward(self, position_hidden: torch.Tensor) -> torch.Tensor:
        if position_hidden.ndim != 3:
            raise ValueError("position_hidden must have shape [batch, sequence, hidden]")
        state_hidden = position_hidden[:, self.position_layout.state_element_index]
        square_start = self.position_layout.square_element_offset
        square_end = square_start + self.position_layout.square_element_count
        square_hidden = position_hidden[:, square_start:square_end]
        if square_hidden.size(1) != self.square_count:
            raise ValueError("position_hidden square span does not match the action-plane square count")
        state_logits = self.state_action_scorer(state_hidden)
        square_logits = self.square_move_type_scorer(square_hidden).reshape(position_hidden.size(0), -1)
        return state_logits + square_logits + self.action_logit_bias
