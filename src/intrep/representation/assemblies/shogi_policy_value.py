from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.shogi_position_modules import (
    SHOGI_RICH_POSITION_HIDDEN_LAYOUT,
    build_shogi_position_encoder as build_shogi_position_encoder_for_module,
    shogi_position_hidden_layout,
)
from intrep.representation.shogi_position_hidden import ShogiPositionHiddenLayout
from intrep.representation.outputs.scalar_value import ScalarTanhValueHead
from intrep.representation.outputs.shogi_legal_move import (
    ShogiLegalMoveAttentionPolicyOutput,
    ShogiStateSummaryLegalMovePolicyOutput,
)
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_RICH_POSITION_INPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    shogi_policy_value_assembly_spec_for_id,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.outputs.shogi_policy_plane_encoding import SHOGI_POLICY_PLANE_ACTION_COUNT


@dataclass(frozen=True)
class SharedCoreShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyPlaneShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    num_heads: int = 8
    hidden_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.0


class SharedCoreShogiPolicyValueModel(nn.Module):
    def __init__(
        self,
        config: SharedCoreShogiPolicyValueModelConfig | None = None,
        *,
        encoder: nn.Module | None = None,
        policy_output: ShogiLegalMoveAttentionPolicyOutput | ShogiStateSummaryLegalMovePolicyOutput | None = None,
        value_output: ScalarTanhValueHead | None = None,
        position_layout: ShogiPositionHiddenLayout | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SharedCoreShogiPolicyValueModelConfig()
        self.position_layout = position_layout or SHOGI_RICH_POSITION_HIDDEN_LAYOUT
        self.encoder = encoder or _build_shogi_position_encoder(
            embedding_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.policy_output = policy_output or _build_legal_move_policy_output(self.config, self.position_layout)
        self.value_output = value_output or ScalarTanhValueHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_feature_ids: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.policy_output(
            position_hidden=position_hidden,
            legal_move_feature_ids=legal_move_feature_ids,
            legal_move_mask=legal_move_mask,
        )

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_feature_ids: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_element_hidden(position_hidden, self.position_layout)
        logits = self.policy_output(
            position_hidden=position_hidden,
            legal_move_feature_ids=legal_move_feature_ids,
            legal_move_mask=legal_move_mask,
        )
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_element_hidden(position_hidden, self.position_layout))


class PolicyPlaneShogiPolicyValueModel(nn.Module):
    def __init__(
        self,
        config: PolicyPlaneShogiPolicyValueModelConfig | None = None,
        *,
        encoder: nn.Module | None = None,
        policy_output: ShogiPolicyPlaneHead | None = None,
        value_output: ScalarTanhValueHead | None = None,
        position_layout: ShogiPositionHiddenLayout | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PolicyPlaneShogiPolicyValueModelConfig()
        self.position_layout = position_layout or SHOGI_RICH_POSITION_HIDDEN_LAYOUT
        self.encoder = encoder or _build_shogi_position_encoder(
            embedding_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.policy_output = policy_output or ShogiPolicyPlaneHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
            action_count=SHOGI_POLICY_PLANE_ACTION_COUNT,
        )
        self.value_output = value_output or ScalarTanhValueHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(self, position_features: ShogiPositionFeatures, policy_plane_legal_mask: torch.Tensor) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_element_hidden(position_hidden, self.position_layout)
        return self.policy_output(position_embedding, policy_plane_legal_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        policy_plane_legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_element_hidden(position_hidden, self.position_layout)
        logits = self.policy_output(position_embedding, policy_plane_legal_mask)
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_element_hidden(position_hidden, self.position_layout))


def _state_element_hidden(
    position_hidden: torch.Tensor,
    position_layout: ShogiPositionHiddenLayout = SHOGI_RICH_POSITION_HIDDEN_LAYOUT,
) -> torch.Tensor:
    return position_hidden[:, position_layout.state_element_index]


def _build_shogi_policy_value_model_from_policy_output(
    *,
    position_input: str,
    policy_output: str,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Module:
    position_layout = shogi_position_hidden_layout(position_input)
    if policy_output in (
        SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
        SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    ):
        shared_config = SharedCoreShogiPolicyValueModelConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        return SharedCoreShogiPolicyValueModel(
            shared_config,
            encoder=_build_shogi_position_encoder(
                input_module_id=position_input,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ),
            policy_output=_build_legal_move_policy_output_from_id(policy_output, shared_config, position_layout),
            position_layout=position_layout,
        )
    if policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ),
            encoder=_build_shogi_position_encoder(
                input_module_id=position_input,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            ),
            position_layout=position_layout,
        )
    raise ValueError(f"unsupported shogi policy/value policy output: {policy_output}")


def build_shogi_policy_value_model_for_assembly_spec(
    *,
    assembly_spec_id: str,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Module:
    assembly_spec = shogi_policy_value_assembly_spec_for_id(assembly_spec_id)
    return _build_shogi_policy_value_model_from_policy_output(
        position_input=str(assembly_spec["input"]),
        policy_output=str(assembly_spec["policy_output"]),
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def _build_shogi_position_encoder(
    *,
    input_module_id: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    core = SharedTransformerCore(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    return build_shogi_position_encoder_for_module(
        input_module_id=input_module_id,
        embedding_dim=embedding_dim,
        core=core,
    )


def _build_legal_move_policy_output(
    config: SharedCoreShogiPolicyValueModelConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> ShogiLegalMoveAttentionPolicyOutput:
    return ShogiLegalMoveAttentionPolicyOutput(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        position_layout=position_layout,
    )


def _build_legal_move_policy_output_from_id(
    policy_output: str,
    config: SharedCoreShogiPolicyValueModelConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> ShogiLegalMoveAttentionPolicyOutput | ShogiStateSummaryLegalMovePolicyOutput:
    if policy_output == SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID:
        return _build_legal_move_policy_output(config, position_layout)
    if policy_output == SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID:
        return _build_state_summary_legal_move_policy_output(config, position_layout)
    raise ValueError(f"unsupported shogi legal move policy output: {policy_output}")


def _build_state_summary_legal_move_policy_output(
    config: SharedCoreShogiPolicyValueModelConfig,
    position_layout: ShogiPositionHiddenLayout,
) -> ShogiStateSummaryLegalMovePolicyOutput:
    return ShogiStateSummaryLegalMovePolicyOutput(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        position_layout=position_layout,
    )
