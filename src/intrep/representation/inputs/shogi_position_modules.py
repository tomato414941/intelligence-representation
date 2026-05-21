from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

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
from intrep.representation.inputs.shogi_minimal_single_global_position import (
    ShogiMinimalSingleGlobalPositionAttentionLogitBias,
    ShogiMinimalSingleGlobalPositionEncoder,
    ShogiMinimalSingleGlobalPositionInputLayer,
)
from intrep.representation.inputs.shogi_minimal_split_global_position import (
    ShogiMinimalSplitGlobalPositionAttentionLogitBias,
    ShogiMinimalSplitGlobalPositionEncoder,
    ShogiMinimalSplitGlobalPositionInputLayer,
)
from intrep.representation.inputs.shogi_position_module_ids import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_RICH_POSITION_INPUT_MODULE_ID,
)
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
from intrep.representation.inputs.shogi_rich_position import (
    ShogiRichPositionAttentionLogitBias,
    ShogiRichPositionEncoder,
    ShogiRichPositionInputLayer,
)
from intrep.representation.shogi_position_hidden import ShogiPositionHiddenLayout


SHOGI_RICH_POSITION_HIDDEN_LAYOUT = ShogiPositionHiddenLayout(
    state_element_index=STATE_ELEMENT_INDEX,
    square_element_offset=RICH_SQUARE_ELEMENT_OFFSET,
    square_element_count=SHOGI_POSITION_SQUARE_COUNT,
)


@dataclass(frozen=True)
class ShogiPositionInputModule:
    module_id: str
    hidden_layout: ShogiPositionHiddenLayout
    input_layer_type: Callable[..., nn.Module]
    attention_logit_bias_type: Callable[..., nn.Module]
    encoder_type: Callable[..., nn.Module]

    def build_encoder(self, *, embedding_dim: int, core: nn.Module) -> nn.Module:
        return self.encoder_type(
            input_layer=self.input_layer_type(embedding_dim=embedding_dim),
            attention_logit_bias=self.attention_logit_bias_type(),
            core=core,
        )


_SHOGI_POSITION_INPUT_MODULES_BY_ID = {
    SHOGI_RICH_POSITION_INPUT_MODULE_ID: ShogiPositionInputModule(
        module_id=SHOGI_RICH_POSITION_INPUT_MODULE_ID,
        hidden_layout=SHOGI_RICH_POSITION_HIDDEN_LAYOUT,
        input_layer_type=ShogiRichPositionInputLayer,
        attention_logit_bias_type=ShogiRichPositionAttentionLogitBias,
        encoder_type=ShogiRichPositionEncoder,
    ),
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID: ShogiPositionInputModule(
        module_id=SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
        hidden_layout=ShogiPositionHiddenLayout(
            state_element_index=SHOGI_ALPHA_ZERO_LIKE_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_ALPHA_ZERO_LIKE_SQUARE_ELEMENT_COUNT,
        ),
        input_layer_type=ShogiAlphaZeroLikePositionInputLayer,
        attention_logit_bias_type=ShogiAlphaZeroLikePositionAttentionLogitBias,
        encoder_type=ShogiAlphaZeroLikePositionEncoder,
    ),
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID: ShogiPositionInputModule(
        module_id=SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID,
        hidden_layout=ShogiPositionHiddenLayout(
            state_element_index=SHOGI_DLSHOGI_LIKE_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT,
        ),
        input_layer_type=ShogiDlshogiLikePositionInputLayer,
        attention_logit_bias_type=ShogiDlshogiLikePositionAttentionLogitBias,
        encoder_type=ShogiDlshogiLikePositionEncoder,
    ),
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID: ShogiPositionInputModule(
        module_id=SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID,
        hidden_layout=ShogiPositionHiddenLayout(
            state_element_index=SHOGI_MINIMAL_SINGLE_GLOBAL_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT,
        ),
        input_layer_type=ShogiMinimalSingleGlobalPositionInputLayer,
        attention_logit_bias_type=ShogiMinimalSingleGlobalPositionAttentionLogitBias,
        encoder_type=ShogiMinimalSingleGlobalPositionEncoder,
    ),
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID: ShogiPositionInputModule(
        module_id=SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID,
        hidden_layout=ShogiPositionHiddenLayout(
            state_element_index=SHOGI_MINIMAL_SPLIT_GLOBAL_STATE_ELEMENT_INDEX,
            square_element_offset=SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET,
            square_element_count=SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT,
        ),
        input_layer_type=ShogiMinimalSplitGlobalPositionInputLayer,
        attention_logit_bias_type=ShogiMinimalSplitGlobalPositionAttentionLogitBias,
        encoder_type=ShogiMinimalSplitGlobalPositionEncoder,
    ),
}


def shogi_position_input_module(input_module_id: str) -> ShogiPositionInputModule:
    if input_module_id not in _SHOGI_POSITION_INPUT_MODULES_BY_ID:
        raise ValueError(f"unsupported shogi position input module: {input_module_id}")
    return _SHOGI_POSITION_INPUT_MODULES_BY_ID[input_module_id]


def shogi_position_hidden_layout(input_module_id: str) -> ShogiPositionHiddenLayout:
    return shogi_position_input_module(input_module_id).hidden_layout


def build_shogi_position_encoder(
    *,
    input_module_id: str = SHOGI_RICH_POSITION_INPUT_MODULE_ID,
    embedding_dim: int,
    core: nn.Module,
) -> nn.Module:
    return shogi_position_input_module(input_module_id).build_encoder(
        embedding_dim=embedding_dim,
        core=core,
    )
