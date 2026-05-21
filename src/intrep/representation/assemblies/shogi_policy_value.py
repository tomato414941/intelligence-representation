from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from intrep.representation.cores.transformer import SharedTransformerCore
from intrep.representation.inputs.shogi_rich_position import (
    ShogiRichPositionEncoder,
    ShogiRichPositionAttentionLogitBias,
    ShogiRichPositionInputLayer,
)
from intrep.representation.inputs.shogi_alpha_zero_like_position import (
    ShogiAlphaZeroLikePositionAttentionLogitBias,
    ShogiAlphaZeroLikePositionEncoder,
    ShogiAlphaZeroLikePositionInputLayer,
)
from intrep.representation.inputs.shogi_dlshogi_like_position import (
    ShogiDlshogiLikePositionAttentionLogitBias,
    ShogiDlshogiLikePositionEncoder,
    ShogiDlshogiLikePositionInputLayer,
)
from intrep.representation.inputs.shogi_minimal_split_global_position import (
    ShogiMinimalSplitGlobalPositionAttentionLogitBias,
    ShogiMinimalSplitGlobalPositionEncoder,
    ShogiMinimalSplitGlobalPositionInputLayer,
)
from intrep.representation.inputs.shogi_minimal_single_global_position import (
    ShogiMinimalSingleGlobalPositionAttentionLogitBias,
    ShogiMinimalSingleGlobalPositionEncoder,
    ShogiMinimalSingleGlobalPositionInputLayer,
)
from intrep.representation.shogi_position_hidden import ShogiPositionHiddenLayout
from intrep.representation.outputs.scalar_value import ScalarTanhValueHead
from intrep.representation.outputs.shogi_legal_move import (
    ShogiLegalMoveAttentionPolicyOutput,
    ShogiStateSummaryLegalMovePolicyOutput,
)
from intrep.representation.outputs.shogi_policy_plane import ShogiPolicyPlaneHead
from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_RICH_POSITION_INPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    shogi_policy_value_assembly_spec_for_id,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_alpha_zero_like import (
    SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_COUNT,
    SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_OFFSET,
    SHOGI_ALPHA_ZERO_LIKE_STATE_ELEMENT_INDEX,
)
from intrep.representation.inputs.shogi_position_features.position_dlshogi_like import (
    SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT,
    SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
    SHOGI_DLSHOGI_LIKE_STATE_ELEMENT_INDEX,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT,
    SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET,
    SHOGI_MINIMAL_SINGLE_GLOBAL_STATE_ELEMENT_INDEX,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_split_global import (
    SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT,
    SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET,
    SHOGI_MINIMAL_SPLIT_GLOBAL_STATE_ELEMENT_INDEX,
)
from intrep.representation.inputs.shogi_position_features.position_schema import (
    RICH_SQUARE_ELEMENT_OFFSET,
    SHOGI_POSITION_SQUARE_COUNT,
    STATE_ELEMENT_INDEX,
)
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


SHOGI_RICH_POSITION_HIDDEN_LAYOUT = ShogiPositionHiddenLayout(
    state_element_index=STATE_ELEMENT_INDEX,
    square_element_offset=RICH_SQUARE_ELEMENT_OFFSET,
    square_element_count=SHOGI_POSITION_SQUARE_COUNT,
)


class SharedCoreShogiPolicyValueModel(nn.Module):
    def __init__(
        self,
        config: SharedCoreShogiPolicyValueModelConfig | None = None,
        *,
        encoder: ShogiRichPositionEncoder | None = None,
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
        encoder: ShogiRichPositionEncoder | None = None,
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
    position_layout = _shogi_position_hidden_layout(position_input)
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
) -> (
    ShogiRichPositionEncoder
    | ShogiAlphaZeroLikePositionEncoder
    | ShogiDlshogiLikePositionEncoder
    | ShogiMinimalSingleGlobalPositionEncoder
    | ShogiMinimalSplitGlobalPositionEncoder
):
    core = SharedTransformerCore(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )
    if input_module_id == SHOGI_RICH_POSITION_INPUT_MODULE_ID:
        return ShogiRichPositionEncoder(
            input_layer=ShogiRichPositionInputLayer(embedding_dim=embedding_dim),
            attention_logit_bias=ShogiRichPositionAttentionLogitBias(),
            core=core,
        )
    if input_module_id == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return ShogiAlphaZeroLikePositionEncoder(
            input_layer=ShogiAlphaZeroLikePositionInputLayer(embedding_dim=embedding_dim),
            attention_logit_bias=ShogiAlphaZeroLikePositionAttentionLogitBias(),
            core=core,
        )
    if input_module_id == SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID:
        return ShogiDlshogiLikePositionEncoder(
            input_layer=ShogiDlshogiLikePositionInputLayer(embedding_dim=embedding_dim),
            attention_logit_bias=ShogiDlshogiLikePositionAttentionLogitBias(),
            core=core,
        )
    if input_module_id == SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID:
        return ShogiMinimalSingleGlobalPositionEncoder(
            input_layer=ShogiMinimalSingleGlobalPositionInputLayer(embedding_dim=embedding_dim),
            attention_logit_bias=ShogiMinimalSingleGlobalPositionAttentionLogitBias(),
            core=core,
        )
    if input_module_id == SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID:
        return ShogiMinimalSplitGlobalPositionEncoder(
            input_layer=ShogiMinimalSplitGlobalPositionInputLayer(embedding_dim=embedding_dim),
            attention_logit_bias=ShogiMinimalSplitGlobalPositionAttentionLogitBias(),
            core=core,
        )
    raise ValueError(f"unsupported shogi position input module: {input_module_id}")


def _shogi_position_hidden_layout(input_module_id: str) -> ShogiPositionHiddenLayout:
    if input_module_id == SHOGI_RICH_POSITION_INPUT_MODULE_ID:
        return SHOGI_RICH_POSITION_HIDDEN_LAYOUT
    if input_module_id == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return ShogiPositionHiddenLayout(
            state_element_index=SHOGI_ALPHA_ZERO_LIKE_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_COUNT,
        )
    if input_module_id == SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID:
        return ShogiPositionHiddenLayout(
            state_element_index=SHOGI_DLSHOGI_LIKE_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT,
        )
    if input_module_id == SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID:
        return ShogiPositionHiddenLayout(
            state_element_index=SHOGI_MINIMAL_SINGLE_GLOBAL_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT,
        )
    if input_module_id == SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID:
        return ShogiPositionHiddenLayout(
            state_element_index=SHOGI_MINIMAL_SPLIT_GLOBAL_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT,
        )
    raise ValueError(f"unsupported shogi position input module: {input_module_id}")


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
