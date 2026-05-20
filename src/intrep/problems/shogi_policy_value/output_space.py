from __future__ import annotations

from intrep.representation.assembly_specs.shogi_policy_value import (
    SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
    SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
)

SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE = "legal_move"
SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE = "policy_plane"
SHOGI_POLICY_VALUE_OUTPUT_SPACES = (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
)


def shogi_policy_value_output_space_for_policy_output(policy_output: str) -> str:
    if policy_output in (
        SHOGI_LEGAL_MOVE_ATTENTION_POLICY_OUTPUT_MODULE_ID,
        SHOGI_STATE_SUMMARY_LEGAL_MOVE_POLICY_OUTPUT_MODULE_ID,
    ):
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE
    if policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE
    raise ValueError(f"unsupported shogi policy/value policy output: {policy_output}")


def validate_shogi_policy_value_output_space(output_space: str) -> None:
    if output_space not in SHOGI_POLICY_VALUE_OUTPUT_SPACES:
        raise ValueError(f"unsupported shogi policy/value output space: {output_space}")
