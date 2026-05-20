from __future__ import annotations


SHOGI_POLICY_VALUE_ASSEMBLY_ID = "shogi_policy_value"
SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID = (
    "shogi_policy_value_position_transformer_legal_move_attention"
)
SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID = (
    "shogi_policy_value_position_transformer_state_summary_legal_move"
)
SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID = "shogi_policy_value_position_transformer_policy_plane"
SHOGI_POLICY_VALUE_ASSEMBLY_SPEC_IDS = (
    SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
)
SHOGI_POLICY_VALUE_DEFAULT_ASSEMBLY_SPEC_ID = SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID

SHOGI_POSITION_INPUT_MODULE_ID = (
    "shogi_global_square_piece_line_pair_drop_shadow_coarse_counterfactual_drop_potential_position_input"
)
SHOGI_SHARED_CORE_MODULE_ID = "shared_transformer_core"
SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID = "shogi_legal_move_attention_policy_output"
SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID = "shogi_state_summary_legal_move_policy_output"
SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID = "shogi_policy_plane_policy_output"
SHOGI_VALUE_OUTPUT_MODULE_ID = "scalar_tanh_value_output"

_SHOGI_POLICY_VALUE_ASSEMBLY_SPECS_BY_ID = {
    SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID: {
        "assembly": SHOGI_POLICY_VALUE_ASSEMBLY_ID,
        "assembly_spec_id": SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
    SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID: {
        "assembly": SHOGI_POLICY_VALUE_ASSEMBLY_ID,
        "assembly_spec_id": SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
    SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID: {
        "assembly": SHOGI_POLICY_VALUE_ASSEMBLY_ID,
        "assembly_spec_id": SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
}


def shogi_policy_value_assembly_spec_for_id(assembly_spec_id: str) -> dict[str, object]:
    if assembly_spec_id not in _SHOGI_POLICY_VALUE_ASSEMBLY_SPECS_BY_ID:
        raise ValueError(f"unsupported shogi policy/value assembly spec: {assembly_spec_id}")
    return dict(_SHOGI_POLICY_VALUE_ASSEMBLY_SPECS_BY_ID[assembly_spec_id])


def shogi_policy_value_policy_output_for_assembly_spec_id(assembly_spec_id: str) -> str:
    return str(shogi_policy_value_assembly_spec_for_id(assembly_spec_id)["policy_output"])
