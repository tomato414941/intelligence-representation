from __future__ import annotations

import hashlib
import json

import shogi
import torch

from intrep.representation.inputs.shogi_position_features.position_features import ShogiPairRelationEdges, ShogiPositionFeatures
from intrep.representation.inputs.shogi_position_features.position_schema import (
    HAND_PIECE_TYPES,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_POSITION_STATE_FEATURE_ID,
)
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    hand_feature_ids,
    move_count_bucket_feature_id,
    relative_square_feature_id,
    side_to_move_feature_id,
)


SHOGI_MINIMAL_GLOBAL_POSITION_INPUT_SCHEMA_ID = "shogi_minimal_global_position_features"
SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT = 1
SHOGI_MINIMAL_GLOBAL_GLOBAL_FEATURE_COUNT = 17
SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_COUNT = 81
SHOGI_MINIMAL_GLOBAL_SQUARE_FEATURE_COUNT = 1
SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT = (
    SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT + SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_COUNT
)
SHOGI_MINIMAL_GLOBAL_STATE_ELEMENT_INDEX = 0
SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_OFFSET = SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT


def shogi_minimal_global_position_feature_manifest() -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_MINIMAL_GLOBAL_POSITION_INPUT_SCHEMA_ID,
        "coordinate_system": "side_to_move_relative_180_rotation",
        "representation_element_count": SHOGI_MINIMAL_GLOBAL_ELEMENT_COUNT,
        "global_element_count": SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT,
        "global_feature_count": SHOGI_MINIMAL_GLOBAL_GLOBAL_FEATURE_COUNT,
        "square_element_count": SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_COUNT,
        "square_feature_count": SHOGI_MINIMAL_GLOBAL_SQUARE_FEATURE_COUNT,
        "feature_vocab_size": SHOGI_POSITION_FEATURE_VOCAB_SIZE,
        "feature_groups": ["global", "square"],
        "global_features": [
            "state",
            "side_to_move",
            "move_count_bucket",
            "own_hand_counts",
            "opponent_hand_counts",
        ],
        "square_features": ["piece_identity"],
        "hand_piece_types": list(HAND_PIECE_TYPES),
    }


def shogi_minimal_global_position_feature_manifest_hash() -> str:
    payload = json.dumps(shogi_minimal_global_position_feature_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST = shogi_minimal_global_position_feature_manifest()
SHOGI_MINIMAL_GLOBAL_POSITION_FEATURE_MANIFEST_HASH = shogi_minimal_global_position_feature_manifest_hash()


def shogi_minimal_global_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board = shogi.Board(position_sfen)
    global_feature_ids = torch.tensor(
        [
            [
                SHOGI_POSITION_STATE_FEATURE_ID,
                side_to_move_feature_id(board.turn),
                move_count_bucket_feature_id(board.move_number),
                *hand_feature_ids(board),
            ]
        ],
        dtype=torch.long,
    )
    square_feature_ids = torch.tensor(
        [[relative_square_feature_id(board, relative_square)] for relative_square in range(81)],
        dtype=torch.long,
    )
    return ShogiPositionFeatures(
        global_feature_ids=global_feature_ids,
        square_feature_ids=square_feature_ids,
        piece_feature_ids=torch.empty((0, 0), dtype=torch.long),
        line_feature_ids=torch.empty((0, 0), dtype=torch.long),
        pair_relation_edges=ShogiPairRelationEdges(
            source_element_indices=torch.empty((0,), dtype=torch.long),
            target_element_indices=torch.empty((0,), dtype=torch.long),
            relation_ids=torch.empty((0,), dtype=torch.long),
        ),
    )


def validate_shogi_minimal_global_position_feature_structure(features: ShogiPositionFeatures) -> None:
    _validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_MINIMAL_GLOBAL_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_GLOBAL_GLOBAL_FEATURE_COUNT),
    )
    _validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_MINIMAL_GLOBAL_SQUARE_ELEMENT_COUNT, SHOGI_MINIMAL_GLOBAL_SQUARE_FEATURE_COUNT),
    )
    _validate_integer_tensor_shape("piece_feature_ids", features.piece_feature_ids, (0, 0))
    _validate_integer_tensor_shape("line_feature_ids", features.line_feature_ids, (0, 0))
    if int(features.pair_relation_edges.relation_ids.numel()) != 0:
        raise ValueError("minimal-global position features must not contain pair relation edges")


def _validate_integer_tensor_shape(name: str, tensor: object, expected_shape: tuple[int, ...]) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if not tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must use an integer dtype")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
