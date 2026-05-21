from __future__ import annotations

from collections.abc import Callable

from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_POSITION_INPUT_MODULE_ID,
    shogi_policy_value_input_for_assembly_spec_id,
)
from intrep.worlds.shogi.position_alpha_zero_like import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST,
    SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_SCHEMA_ID,
    shogi_alpha_zero_like_position_features_from_sfen,
)
from intrep.worlds.shogi.position_encoding import (
    SHOGI_POSITION_FEATURE_MANIFEST,
    SHOGI_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_POSITION_INPUT_SCHEMA_ID,
    ShogiPositionFeatures,
    shogi_position_features_from_sfen,
)


ShogiPositionFeatureBuilder = Callable[[str], ShogiPositionFeatures]


def shogi_position_feature_builder_for_assembly_spec_id(assembly_spec_id: str) -> ShogiPositionFeatureBuilder:
    return shogi_position_feature_builder_for_input_module(
        shogi_policy_value_input_for_assembly_spec_id(assembly_spec_id)
    )


def shogi_position_feature_builder_for_input_module(input_module: str) -> ShogiPositionFeatureBuilder:
    if input_module == SHOGI_POSITION_INPUT_MODULE_ID:
        return shogi_position_features_from_sfen
    if input_module == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return shogi_alpha_zero_like_position_features_from_sfen
    raise ValueError(f"unsupported shogi position input module: {input_module}")


def shogi_position_input_identity_for_assembly_spec_id(assembly_spec_id: str) -> dict[str, object]:
    return shogi_position_input_identity_for_input_module(
        shogi_policy_value_input_for_assembly_spec_id(assembly_spec_id)
    )


def shogi_position_input_identity_for_input_module(input_module: str) -> dict[str, object]:
    if input_module == SHOGI_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_POSITION_FEATURE_MANIFEST_HASH,
        }
    if input_module == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST_HASH,
        }
    raise ValueError(f"unsupported shogi position input module: {input_module}")
