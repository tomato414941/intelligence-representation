from __future__ import annotations

from intrep.representation.assembly_specs.shogi_policy_value import shogi_policy_value_policy_output_for_assembly_spec_id
from intrep.representation.outputs.shogi_policy_outputs import (
    ShogiPolicyOutputKind,
    shogi_policy_output_kind,
)

SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE = "legal_move"
SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE = "policy_plane"
SHOGI_POLICY_VALUE_OUTPUT_SPACES = (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
)


def shogi_policy_value_output_space_for_policy_output(policy_output: str) -> str:
    policy_output_kind = shogi_policy_output_kind(policy_output)
    if policy_output_kind == ShogiPolicyOutputKind.LEGAL_MOVE:
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_LEGAL_MOVE
    if policy_output_kind == ShogiPolicyOutputKind.POLICY_PLANE:
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE
    raise ValueError(f"unsupported shogi policy/value policy output: {policy_output}")


def shogi_policy_value_output_space_for_assembly_spec(assembly_spec_id: str) -> str:
    return shogi_policy_value_output_space_for_policy_output(
        shogi_policy_value_policy_output_for_assembly_spec_id(assembly_spec_id)
    )


def validate_shogi_policy_value_output_space(output_space: str) -> None:
    if output_space not in SHOGI_POLICY_VALUE_OUTPUT_SPACES:
        raise ValueError(f"unsupported shogi policy/value output space: {output_space}")
