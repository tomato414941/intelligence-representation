from __future__ import annotations

import hashlib
import json

import torch

from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPositionFeatures,
    validate_empty_pair_relation_edges,
    validate_integer_tensor_shape,
)
from intrep.representation.inputs.shogi_position_features.position_sfen import (
    SHOGI_EMPTY_FEATURE_IDS,
    SHOGI_EMPTY_PAIR_RELATION_EDGES,
    shogi_hand_feature_ids_from_sfen,
    shogi_square_piece_feature_id_rows_from_sfen,
    split_shogi_sfen,
)
from intrep.representation.inputs.shogi_position_features.position_schema import (
    HAND_PIECE_TYPES,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_POSITION_STATE_FEATURE_ID,
    SIDE_TO_MOVE_BLACK_FEATURE_ID,
    SIDE_TO_MOVE_WHITE_FEATURE_ID,
)
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    move_count_bucket_feature_id,
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
    board_sfen, turn_sfen, hand_sfen, move_number_sfen = split_shogi_sfen(position_sfen)
    own_is_black = turn_sfen == "b"
    global_feature_ids = torch.tensor(
        [
            [
                SHOGI_POSITION_STATE_FEATURE_ID,
                SIDE_TO_MOVE_BLACK_FEATURE_ID if own_is_black else SIDE_TO_MOVE_WHITE_FEATURE_ID,
                move_count_bucket_feature_id(int(move_number_sfen)),
                *shogi_hand_feature_ids_from_sfen(hand_sfen, own_is_black=own_is_black),
            ]
        ],
        dtype=torch.long,
    )
    return ShogiPositionFeatures(
        global_feature_ids=global_feature_ids,
        square_feature_ids=torch.tensor(
            shogi_square_piece_feature_id_rows_from_sfen(board_sfen, own_is_black=own_is_black),
            dtype=torch.long,
        ),
        piece_feature_ids=SHOGI_EMPTY_FEATURE_IDS,
        line_feature_ids=SHOGI_EMPTY_FEATURE_IDS,
        pair_relation_edges=SHOGI_EMPTY_PAIR_RELATION_EDGES,
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
