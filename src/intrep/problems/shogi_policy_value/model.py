from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.shogi_position import (
    ShogiPositionEncoder,
    ShogiPositionGeometryAttentionBias,
    ShogiPositionInputLayer,
)
from intrep.representation.outputs.scalar_value import ScalarTanhValueHead
from intrep.representation.outputs.shogi_legal_move_token import (
    ShogiDirectCandidateMovePolicyHead,
    ShogiLegalMoveTokenInputLayer,
    ShogiLegalMoveTokenPolicyOutput,
)
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.worlds.shogi.position_encoding import STATE_TOKEN_INDEX, ShogiPositionFeatures
from intrep.worlds.shogi.policy_plane import SHOGI_POLICY_PLANE_ACTION_COUNT


SHOGI_POLICY_VALUE_MODEL_ID = "shogi_policy_value"
SHOGI_POSITION_INPUT_MODULE_ID = (
    "shogi_global_square_piece_line_pair_drop_shadow_coarse_counterfactual_drop_potential_position_input"
)
SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID = "shogi_direct_candidate_move_policy_output"
SHOGI_SHARED_CORE_MODULE_ID = "shared_transformer_core_with_shogi_pair_relation_bias"
SHOGI_NO_CORE_MODULE_ID = "none"
SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID = "shogi_legal_move_token_policy_output"
SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID = "shogi_policy_plane_policy_output"
SHOGI_VALUE_OUTPUT_MODULE_ID = "scalar_tanh_value_output"
SHOGI_POSITION_INPUT_MODULE_IDS = (SHOGI_POSITION_INPUT_MODULE_ID,)
SHOGI_CORE_MODULE_IDS = (SHOGI_SHARED_CORE_MODULE_ID, SHOGI_NO_CORE_MODULE_ID)
SHOGI_POLICY_OUTPUT_MODULE_IDS = (
    SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID,
)
SHOGI_VALUE_OUTPUT_MODULE_IDS = (SHOGI_VALUE_OUTPUT_MODULE_ID,)


def shogi_policy_value_model_spec(
    *,
    input: str,
    core: str,
    policy_output: str,
    value_output: str,
) -> dict[str, object]:
    validate_shogi_policy_value_components(
        input=input,
        core=core,
        policy_output=policy_output,
        value_output=value_output,
    )
    return {
        "model": SHOGI_POLICY_VALUE_MODEL_ID,
        "input": input,
        "core": core,
        "policy_output": policy_output,
        "value_output": value_output,
    }


def validate_shogi_policy_value_components(
    *,
    input: str,
    core: str,
    policy_output: str,
    value_output: str,
) -> None:
    if input not in SHOGI_POSITION_INPUT_MODULE_IDS:
        raise ValueError(f"unsupported shogi policy/value input: {input}")
    if core not in SHOGI_CORE_MODULE_IDS:
        raise ValueError(f"unsupported shogi policy/value core: {core}")
    if policy_output not in SHOGI_POLICY_OUTPUT_MODULE_IDS:
        raise ValueError(f"unsupported shogi policy/value policy output: {policy_output}")
    if value_output not in SHOGI_VALUE_OUTPUT_MODULE_IDS:
        raise ValueError(f"unsupported shogi policy/value value output: {value_output}")
    if policy_output == SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID and core != SHOGI_NO_CORE_MODULE_ID:
        raise ValueError("direct candidate-move policy output requires core=none")
    if policy_output != SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID and core == SHOGI_NO_CORE_MODULE_ID:
        raise ValueError(f"{policy_output} requires a shared core")


@dataclass(frozen=True)
class DirectCandidateMoveShogiPolicyValueModelConfig:
    embedding_dim: int = 256
    hidden_dim: int = 1024


class DirectCandidateMoveShogiPolicyValueModel(nn.Module):
    def __init__(
        self,
        config: DirectCandidateMoveShogiPolicyValueModelConfig | None = None,
        *,
        input_layer: ShogiPositionInputLayer | None = None,
        move_input: ShogiLegalMoveTokenInputLayer | None = None,
        policy_output: ShogiDirectCandidateMovePolicyHead | None = None,
        value_output: ScalarTanhValueHead | None = None,
    ) -> None:
        super().__init__()
        self.config = config or DirectCandidateMoveShogiPolicyValueModelConfig()
        embedding_dim = self.config.embedding_dim
        self.input = input_layer or ShogiPositionInputLayer(embedding_dim=embedding_dim)
        self.move_input = move_input or ShogiLegalMoveTokenInputLayer(embedding_dim=embedding_dim)
        self.policy_output = policy_output or ShogiDirectCandidateMovePolicyHead(
            input_dim=embedding_dim * 2,
            hidden_dim=self.config.hidden_dim,
        )
        self.value_output = value_output or ScalarTanhValueHead(
            embedding_dim=embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_token_features: torch.Tensor,
        legal_move_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_embedding = self.input(position_features).mean(dim=1)
        move_embedding = self.move_input(legal_move_token_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        return self.policy_output(torch.cat((expanded_position, move_embedding), dim=-1), legal_move_token_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_token_features: torch.Tensor,
        legal_move_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_embedding = self.input(position_features).mean(dim=1)
        move_embedding = self.move_input(legal_move_token_features)
        expanded_position = position_embedding[:, None, :].expand(-1, move_embedding.size(1), -1)
        logits = self.policy_output(torch.cat((expanded_position, move_embedding), dim=-1), legal_move_token_mask)
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_embedding = self.input(position_features).mean(dim=1)
        return self.value_output(position_embedding)


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
        policy_output: ShogiLegalMoveTokenPolicyOutput | None = None,
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
        self.policy_output = policy_output or ShogiLegalMoveTokenPolicyOutput(
            embedding_dim=self.config.embedding_dim,
            num_heads=self.config.num_heads,
            hidden_dim=self.config.hidden_dim,
        )
        self.value_output = value_output or ScalarTanhValueHead(
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_token_features: torch.Tensor,
        legal_move_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.policy_output(
            position_hidden=position_hidden,
            legal_move_token_features=legal_move_token_features,
            legal_move_token_mask=legal_move_token_mask,
        )

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        legal_move_token_features: torch.Tensor,
        legal_move_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_token_hidden(position_hidden)
        logits = self.policy_output(
            position_hidden=position_hidden,
            legal_move_token_features=legal_move_token_features,
            legal_move_token_mask=legal_move_token_mask,
        )
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_token_hidden(position_hidden))


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
        position_embedding = _state_token_hidden(position_hidden)
        return self.policy_output(position_embedding, policy_plane_legal_mask)

    def forward_policy_value(
        self,
        position_features: ShogiPositionFeatures,
        policy_plane_legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_hidden = self.encoder(position_features)
        position_embedding = _state_token_hidden(position_hidden)
        logits = self.policy_output(position_embedding, policy_plane_legal_mask)
        return logits, self.value_output(position_embedding)

    def predict_value(self, position_features: ShogiPositionFeatures) -> torch.Tensor:
        position_hidden = self.encoder(position_features)
        return self.value_output(_state_token_hidden(position_hidden))


def _state_token_hidden(position_hidden: torch.Tensor) -> torch.Tensor:
    return position_hidden[:, STATE_TOKEN_INDEX]


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
        attention_bias=ShogiPositionGeometryAttentionBias(),
        core=SharedTransformerCore(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ),
    )
