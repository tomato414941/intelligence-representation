from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.shogi_position import (
    ShogiPositionEncoder,
    ShogiPositionAttentionLogitBias,
    ShogiPositionInputLayer,
)
from intrep.representation.outputs.scalar_value import ScalarTanhValueHead
from intrep.representation.outputs.shogi_legal_move import (
    ShogiLegalMoveAttentionPolicyOutput,
    ShogiStateSummaryLegalMovePolicyOutput,
)
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    shogi_policy_value_assembly_spec_for_id,
)
from intrep.worlds.shogi.position_encoding import STATE_ELEMENT_INDEX, ShogiPositionFeatures
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT


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
        encoder: ShogiPositionEncoder | None = None,
        policy_output: ShogiLegalMoveAttentionPolicyOutput | ShogiStateSummaryLegalMovePolicyOutput | None = None,
        value_output: ScalarTanhValueHead | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SharedCoreShogiPolicyValueModelConfig()
        self.encoder = encoder or _build_shogi_position_encoder(
            embedding_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        self.policy_output = policy_output or _build_legal_move_policy_output(self.config)
        self.value_output = value_output or ScalarTanhValueHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_features: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.policy_output(
            position_hidden=position_hidden,
            legal_move_features=legal_move_features,
            legal_move_mask=legal_move_mask,
        )

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_features: torch.Tensor,
        legal_move_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_element_hidden(position_hidden)
        logits = self.policy_output(
            position_hidden=position_hidden,
            legal_move_features=legal_move_features,
            legal_move_mask=legal_move_mask,
        )
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_element_hidden(position_hidden))


class PolicyPlaneShogiPolicyValueModel(nn.Module):
    def __init__(
        self,
        config: PolicyPlaneShogiPolicyValueModelConfig | None = None,
        *,
        encoder: ShogiPositionEncoder | None = None,
        policy_output: ShogiPolicyPlaneHead | None = None,
        value_output: ScalarTanhValueHead | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PolicyPlaneShogiPolicyValueModelConfig()
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
        position_embedding = _state_element_hidden(position_hidden)
        return self.policy_output(position_embedding, policy_plane_legal_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        policy_plane_legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_element_hidden(position_hidden)
        logits = self.policy_output(position_embedding, policy_plane_legal_mask)
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_element_hidden(position_hidden))


def _state_element_hidden(position_hidden: torch.Tensor) -> torch.Tensor:
    return position_hidden[:, STATE_ELEMENT_INDEX]


def _build_shogi_policy_value_model_from_policy_output(
    *,
    policy_output: str,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float = 0.0,
) -> nn.Module:
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
            policy_output=_build_legal_move_policy_output_from_id(policy_output, shared_config),
        )
    if policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return PolicyPlaneShogiPolicyValueModel(
            PolicyPlaneShogiPolicyValueModelConfig(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
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
        policy_output=str(assembly_spec["policy_output"]),
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )


def _build_shogi_position_encoder(
    *,
    embedding_dim: int,
    num_heads: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> ShogiPositionEncoder:
    return ShogiPositionEncoder(
        input_layer=ShogiPositionInputLayer(embedding_dim=embedding_dim),
        attention_logit_bias=ShogiPositionAttentionLogitBias(),
        core=SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ),
    )


def _build_legal_move_policy_output(
    config: SharedCoreShogiPolicyValueModelConfig,
) -> ShogiLegalMoveAttentionPolicyOutput:
    return ShogiLegalMoveAttentionPolicyOutput(
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
    )


def _build_legal_move_policy_output_from_id(
    policy_output: str,
    config: SharedCoreShogiPolicyValueModelConfig,
) -> ShogiLegalMoveAttentionPolicyOutput | ShogiStateSummaryLegalMovePolicyOutput:
    if policy_output == SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID:
        return _build_legal_move_policy_output(config)
    if policy_output == SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID:
        return _build_state_summary_legal_move_policy_output(config)
    raise ValueError(f"unsupported shogi legal move policy output: {policy_output}")


def _build_state_summary_legal_move_policy_output(
    config: SharedCoreShogiPolicyValueModelConfig,
) -> ShogiStateSummaryLegalMovePolicyOutput:
    return ShogiStateSummaryLegalMovePolicyOutput(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
