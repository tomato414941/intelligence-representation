from __future__ import annotations

import hashlib
import json

import shogi
import torch

from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPairRelationEdges,
    ShogiPositionFeatures,
    validate_empty_pair_relation_edges,
    validate_integer_tensor_shape,
)
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


SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_SCHEMA_ID = "shogi_minimal_single_global_position_features"
SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_ELEMENT_COUNT = 1
SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FIELD_COUNT = 17
SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT = 81
SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_FIELD_COUNT = 1
SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT = (
    SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_ELEMENT_COUNT + SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT
)
SHOGI_MINIMAL_SINGLE_GLOBAL_STATE_ELEMENT_INDEX = 0
SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_OFFSET = SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_ELEMENT_COUNT


def shogi_minimal_single_global_position_feature_manifest() -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_INPUT_SCHEMA_ID,
        "coordinate_system": "side_to_move_relative_180_rotation",
        "representation_element_count": SHOGI_MINIMAL_SINGLE_GLOBAL_ELEMENT_COUNT,
        "global_element_count": SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_ELEMENT_COUNT,
        "global_field_count": SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FIELD_COUNT,
        "square_element_count": SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT,
        "square_field_count": SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_FIELD_COUNT,
        "feature_vocab_size": SHOGI_POSITION_FEATURE_VOCAB_SIZE,
        "feature_groups": ["global", "square"],
        "global_feature_groups": [
            "state",
            "side_to_move",
            "move_count_bucket",
            "own_hand_counts",
            "opponent_hand_counts",
        ],
        "square_feature_groups": ["piece_identity"],
        "hand_piece_types": list(HAND_PIECE_TYPES),
    }


def shogi_minimal_single_global_position_feature_manifest_hash() -> str:
    payload = json.dumps(shogi_minimal_single_global_position_feature_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST = shogi_minimal_single_global_position_feature_manifest()
SHOGI_MINIMAL_SINGLE_GLOBAL_POSITION_FEATURE_MANIFEST_HASH = shogi_minimal_single_global_position_feature_manifest_hash()


def shogi_minimal_single_global_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
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


def validate_shogi_minimal_single_global_position_feature_structure(features: ShogiPositionFeatures) -> None:
    validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_ELEMENT_COUNT, SHOGI_MINIMAL_SINGLE_GLOBAL_GLOBAL_FIELD_COUNT),
    )
    validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_ELEMENT_COUNT, SHOGI_MINIMAL_SINGLE_GLOBAL_SQUARE_FIELD_COUNT),
    )
    validate_integer_tensor_shape("piece_feature_ids", features.piece_feature_ids, (0, 0))
    validate_integer_tensor_shape("line_feature_ids", features.line_feature_ids, (0, 0))
    validate_empty_pair_relation_edges(features.pair_relation_edges, context="minimal-single-global")
