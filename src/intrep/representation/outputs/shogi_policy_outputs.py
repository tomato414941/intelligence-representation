from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from torch import nn

from intrep.representation.outputs.shogi_legal_move import (
    ShogiLegalMoveAttentionPolicyOutput,
    ShogiStateSummaryLegalMovePolicyOutput,
)
from intrep.representation.outputs.shogi_policy_output_module_ids import (
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
)
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.representation.outputs.shogi_policy_plane_encoding import SHOGI_POLICY_PLANE_ACTION_COUNT
from intrep.representation.shogi_position_hidden import ShogiPositionHiddenLayout


class ShogiPolicyOutputKind(str, Enum):
    LEGAL_MOVE = "legal_move"
    POLICY_PLANE = "policy_plane"


@dataclass(frozen=True)
class ShogiPolicyOutputModuleConfig:
    embedding_dim: int
    num_heads: int
    hidden_dim: int


@dataclass(frozen=True)
class ShogiPolicyOutputModule:
    module_id: str
    kind: ShogiPolicyOutputKind
    factory: Callable[[ShogiPolicyOutputModuleConfig, ShogiPositionHiddenLayout], nn.Module]

    def build(self, *, config: ShogiPolicyOutputModuleConfig, position_layout: ShogiPositionHiddenLayout) -> nn.Module:
        return self.factory(config, position_layout)


def _build_legal_move_attention_output(
    config: ShogiPolicyOutputModuleConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> nn.Module:
    return ShogiLegalMoveAttentionPolicyOutput(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        position_layout=position_layout,
    )


def _build_state_summary_legal_move_output(
    config: ShogiPolicyOutputModuleConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> nn.Module:
    return ShogiStateSummaryLegalMovePolicyOutput(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        position_layout=position_layout,
    )


def _build_policy_plane_output(
    config: ShogiPolicyOutputModuleConfig,
    _position_layout: ShogiPositionHiddenLayout,
) -> nn.Module:
    return ShogiPolicyPlaneHead(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        action_count=SHOGI_POLICY_PLANE_ACTION_COUNT,
    )


_SHOGI_POLICY_OUTPUT_MODULES_BY_ID = {
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID: ShogiPolicyOutputModule(
        module_id=SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
        kind=ShogiPolicyOutputKind.LEGAL_MOVE,
        factory=_build_legal_move_attention_output,
    ),
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID: ShogiPolicyOutputModule(
        module_id=SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
        kind=ShogiPolicyOutputKind.LEGAL_MOVE,
        factory=_build_state_summary_legal_move_output,
    ),
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID: ShogiPolicyOutputModule(
        module_id=SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
        kind=ShogiPolicyOutputKind.POLICY_PLANE,
        factory=_build_policy_plane_output,
    ),
}


def shogi_policy_output_module(module_id: str) -> ShogiPolicyOutputModule:
    if module_id not in _SHOGI_POLICY_OUTPUT_MODULES_BY_ID:
        raise ValueError(f"unsupported shogi policy output module: {module_id}")
    return _SHOGI_POLICY_OUTPUT_MODULES_BY_ID[module_id]


def shogi_policy_output_kind(module_id: str) -> ShogiPolicyOutputKind:
    return shogi_policy_output_module(module_id).kind


def build_shogi_policy_output(
    *,
    module_id: str,
    config: ShogiPolicyOutputModuleConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> nn.Module:
    return shogi_policy_output_module(module_id).build(
        config=config,
        position_layout=position_layout,
    )
