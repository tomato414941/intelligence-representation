from __future__ import annotations

from intrep.problems.shogi_policy_value.model import (
    SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID,
    SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID,
    SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID,
)

SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE = "candidate_move"
SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE = "policy_plane"
SHOGI_POLICY_VALUE_OUTPUT_SPACES = (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
)


def shogi_policy_value_output_space_for_policy_output(policy_output: str) -> str:
    if policy_output in (SHOGI_LEGAL_MOVE_TOKEN_POLICY_OUTPUT_MODULE_ID, SHOGI_DIRECT_POLICY_OUTPUT_MODULE_ID):
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE
    if policy_output == SHOGI_POLICY_PLANE_OUTPUT_MODULE_ID:
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE
    raise ValueError(f"unsupported shogi policy/value policy output: {policy_output}")


def validate_shogi_policy_value_output_space(output_space: str) -> None:
    if output_space not in SHOGI_POLICY_VALUE_OUTPUT_SPACES:
        raise ValueError(f"unsupported shogi policy/value output space: {output_space}")
