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
SHOGI_POSITION_INPUT_MODULE_IDS = (SHOGI_POSITION_INPUT_MODULE_ID,)
SHOGI_CORE_MODULE_IDS = (SHOGI_SHARED_CORE_MODULE_ID,)
SHOGI_POLICY_OUTPUT_MODULE_IDS = (
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
)
SHOGI_VALUE_OUTPUT_MODULE_IDS = (SHOGI_VALUE_OUTPUT_MODULE_ID,)

_SHOGI_POLICY_VALUE_SPEC_BY_POLICY_OUTPUT = {
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID: SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID: SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID: SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID,
}
_SHOGI_POLICY_VALUE_COMPONENTS_BY_SPEC_ID = {
    SHOGI_POLICY_VALUE_LEGAL_MOVE_ATTENTION_ASSEMBLY_SPEC_ID: {
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
    SHOGI_POLICY_VALUE_STATE_SUMMARY_LEGAL_MOVE_ASSEMBLY_SPEC_ID: {
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
    SHOGI_POLICY_VALUE_POLICY_PLANE_ASSEMBLY_SPEC_ID: {
        "input": SHOGI_POSITION_INPUT_MODULE_ID,
        "core": SHOGI_SHARED_CORE_MODULE_ID,
        "policy_output": SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
        "value_output": SHOGI_VALUE_OUTPUT_MODULE_ID,
    },
}


def shogi_policy_value_components_for_assembly_spec_id(assembly_spec_id: str) -> dict[str, str]:
    if assembly_spec_id not in _SHOGI_POLICY_VALUE_COMPONENTS_BY_SPEC_ID:
        raise ValueError(f"unsupported shogi policy/value assembly spec: {assembly_spec_id}")
    return dict(_SHOGI_POLICY_VALUE_COMPONENTS_BY_SPEC_ID[assembly_spec_id])


def shogi_policy_value_assembly_spec_for_id(assembly_spec_id: str) -> dict[str, object]:
    components = shogi_policy_value_components_for_assembly_spec_id(assembly_spec_id)
    return shogi_policy_value_assembly_spec(**components)


def shogi_policy_value_policy_output_for_assembly_spec_id(assembly_spec_id: str) -> str:
    return shogi_policy_value_components_for_assembly_spec_id(assembly_spec_id)["policy_output"]


def shogi_policy_value_assembly_spec(
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
        "assembly": SHOGI_POLICY_VALUE_ASSEMBLY_ID,
        "assembly_spec_id": _SHOGI_POLICY_VALUE_SPEC_BY_POLICY_OUTPUT[policy_output],
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
