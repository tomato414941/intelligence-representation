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
    OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
    SQUARE_ATTACK_PIECE_TYPES,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_POSITION_STATE_FEATURE_ID,
)
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    attack_feature_ids,
    hand_feature_ids,
    in_check_feature_id,
    move_count_bucket_feature_id,
    square_piece_type_attack_feature_id_rows,
    square_piece_plane_feature_id_rows,
    side_to_move_feature_id,
)


SHOGI_DLSHOGI_LIKE_POSITION_INPUT_SCHEMA_ID = "shogi_dlshogi_like_position_features"
SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT = 18
SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT = 81
SHOGI_DLSHOGI_LIKE_SQUARE_FIELD_COUNT = 58
SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT = (
    SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT + SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT
)
SHOGI_DLSHOGI_LIKE_STATE_ELEMENT_INDEX = 0
SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_OFFSET = SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT


def shogi_dlshogi_like_position_feature_manifest() -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_DLSHOGI_LIKE_POSITION_INPUT_SCHEMA_ID,
        "coordinate_system": "side_to_move_relative_180_rotation",
        "representation_element_count": SHOGI_DLSHOGI_LIKE_ELEMENT_COUNT,
        "global_element_count": SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT,
        "square_element_count": SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT,
        "square_field_count": SHOGI_DLSHOGI_LIKE_SQUARE_FIELD_COUNT,
        "feature_vocab_size": SHOGI_POSITION_FEATURE_VOCAB_SIZE,
        "feature_groups": ["global", "square"],
        "global_feature_groups": [
            "state",
            "side_to_move",
            "in_check",
            "move_count_bucket",
            "own_hand_counts",
            "opponent_hand_counts",
        ],
        "square_feature_groups": [
            "own_piece_planes",
            "opponent_piece_planes",
            "own_attack_count",
            "opponent_attack_count",
            "own_piece_type_attacks",
            "opponent_piece_type_attacks",
        ],
        "hand_piece_types": list(HAND_PIECE_TYPES),
        "square_piece_types": list(SQUARE_ATTACK_PIECE_TYPES),
        "vocab_offsets": {
            "own_square_piece_type_attack": OWN_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
            "opponent_square_piece_type_attack": OPPONENT_SQUARE_PIECE_TYPE_ATTACK_OFFSET,
        },
    }


def shogi_dlshogi_like_position_feature_manifest_hash() -> str:
    payload = json.dumps(shogi_dlshogi_like_position_feature_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST = shogi_dlshogi_like_position_feature_manifest()
SHOGI_DLSHOGI_LIKE_POSITION_FEATURE_MANIFEST_HASH = shogi_dlshogi_like_position_feature_manifest_hash()


def shogi_dlshogi_like_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board = shogi.Board(position_sfen)
    global_feature_ids = torch.tensor(
        [
            SHOGI_POSITION_STATE_FEATURE_ID,
            side_to_move_feature_id(board.turn),
            in_check_feature_id(board.is_check()),
            move_count_bucket_feature_id(board.move_number),
            *hand_feature_ids(board),
        ],
        dtype=torch.long,
    )
    square_feature_ids = torch.tensor(_dlshogi_like_square_feature_rows(board), dtype=torch.long)
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


def validate_shogi_dlshogi_like_position_feature_structure(features: ShogiPositionFeatures) -> None:
    validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_DLSHOGI_LIKE_GLOBAL_ELEMENT_COUNT,),
    )
    validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT, SHOGI_DLSHOGI_LIKE_SQUARE_FIELD_COUNT),
    )
    validate_integer_tensor_shape("piece_feature_ids", features.piece_feature_ids, (0, 0))
    validate_integer_tensor_shape("line_feature_ids", features.line_feature_ids, (0, 0))
    validate_empty_pair_relation_edges(features.pair_relation_edges, context="dlshogi-like")


def _dlshogi_like_square_feature_rows(board: shogi.Board) -> list[list[int]]:
    piece_rows = square_piece_plane_feature_id_rows(board)
    attacks = attack_feature_ids(board)
    own_piece_type_attacks, opponent_piece_type_attacks = square_piece_type_attack_feature_id_rows(board)
    rows: list[list[int]] = []
    for relative_square in range(SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT):
        rows.append(
            [
                *piece_rows[relative_square],
                attacks[relative_square],
                attacks[SHOGI_DLSHOGI_LIKE_SQUARE_ELEMENT_COUNT + relative_square],
                *own_piece_type_attacks[relative_square],
                *opponent_piece_type_attacks[relative_square],
            ]
        )
    return rows
