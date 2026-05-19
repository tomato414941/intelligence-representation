from __future__ import annotations

from intrep.problems.shogi_policy_value.model import (
    SHOGI_POLICY_VALUE_MODEL_DIRECT,
    SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER,
    SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER,
)

SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE = "candidate_move"
SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE = "policy_plane"
SHOGI_POLICY_VALUE_OUTPUT_SPACES = (
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE,
    SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE,
)


def shogi_policy_value_output_space_for_model(model_name: str) -> str:
    if model_name in (SHOGI_POLICY_VALUE_MODEL_SHARED_TRANSFORMER, SHOGI_POLICY_VALUE_MODEL_DIRECT):
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_CANDIDATE_MOVE
    if model_name == SHOGI_POLICY_VALUE_MODEL_POLICY_PLANE_SHARED_TRANSFORMER:
        return SHOGI_POLICY_VALUE_OUTPUT_SPACE_POLICY_PLANE
    raise ValueError(f"unsupported shogi policy/value model: {model_name}")


def validate_shogi_policy_value_output_space(output_space: str) -> None:
    if output_space not in SHOGI_POLICY_VALUE_OUTPUT_SPACES:
        raise ValueError(f"unsupported shogi policy/value output space: {output_space}")
