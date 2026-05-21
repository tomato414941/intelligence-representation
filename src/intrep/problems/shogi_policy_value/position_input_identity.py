from __future__ import annotations

from collections.abc import Callable

from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID,
    SHOGI_RICH_POSITION_INPUT_MODULE_ID,
    shogi_policy_value_input_for_assembly_spec_id,
)
from intrep.representation.inputs.shogi_position_features.position_alpha_zero_like import (
    SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST,
    SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_SCHEMA_ID,
    shogi_alpha_zero_like_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_dlshogi_like import (
    SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST,
    SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_DLSHOGI_LIKE_POSITION_INPUT_SCHEMA_ID,
    shogi_dlshogi_like_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_rich import (
    SHOGI_RICH_POSITION_FEATURE_MANIFEST,
    SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_RICH_POSITION_INPUT_SCHEMA_ID,
    shogi_rich_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_features import ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_minimal_split_global import (
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_SCHEMA_ID,
    shogi_minimal_split_global_position_features_from_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_minimal_single_global import (
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
    SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_SCHEMA_ID,
    shogi_minimal_single_global_position_features_from_sfen,
)


ShogiPositionFeatureBuilder = Callable[[str], ShogiPositionFeatures]


def shogi_position_feature_builder_for_assembly_spec_id(assembly_spec_id: str) -> ShogiPositionFeatureBuilder:
    return shogi_position_feature_builder_for_input_module(
        shogi_policy_value_input_for_assembly_spec_id(assembly_spec_id)
    )


def shogi_position_feature_builder_for_input_module(input_module: str) -> ShogiPositionFeatureBuilder:
    if input_module == SHOGI_RICH_POSITION_INPUT_MODULE_ID:
        return shogi_rich_position_features_from_sfen
    if input_module == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return shogi_alpha_zero_like_position_features_from_sfen
    if input_module == SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID:
        return shogi_dlshogi_like_position_features_from_sfen
    if input_module == SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID:
        return shogi_minimal_single_global_position_features_from_sfen
    if input_module == SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID:
        return shogi_minimal_split_global_position_features_from_sfen
    raise ValueError(f"unsupported shogi position input module: {input_module}")


def shogi_position_input_identity_for_assembly_spec_id(assembly_spec_id: str) -> dict[str, object]:
    return shogi_position_input_identity_for_input_module(
        shogi_policy_value_input_for_assembly_spec_id(assembly_spec_id)
    )


def shogi_position_input_identity_for_input_module(input_module: str) -> dict[str, object]:
    if input_module == SHOGI_RICH_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_RICH_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_RICH_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_RICH_POSITION_FEATURE_MANIFEST_HASH,
        }
    if input_module == SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_ALPHA_ZERO_LIKE_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_ALPHA_ZERO_LIKE_POSITION_FEATURE_MANIFEST_HASH,
        }
    if input_module == SHOGI_DLSHOGI_LIKE_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_DLSHOGI_LIKE_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST_HASH,
        }
    if input_module == SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
        }
    if input_module == SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_MODULE_ID:
        return {
            "input_schema_id": SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_SCHEMA_ID,
            "input_feature_manifest": SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST,
            "input_feature_manifest_hash": SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST_HASH,
        }
    raise ValueError(f"unsupported shogi position input module: {input_module}")
