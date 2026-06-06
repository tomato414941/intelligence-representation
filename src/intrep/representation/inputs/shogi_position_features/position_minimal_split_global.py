from __future__ import annotations

import hashlib
import json

import torch

from intrep.representation.inputs.shogi_position_features.position_features import (
    ShogiPairRelationEdges,
    ShogiPositionFeatures,
    validate_empty_pair_relation_edges,
    validate_integer_tensor_shape,
)
from intrep.representation.inputs.shogi_position_features.position_schema import (
    EMPTY_SQUARE_FEATURE_ID,
    HAND_PIECE_TYPES,
    HAND_COUNT_BUCKET_MAX,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    SHOGI_POSITION_FEATURE_VOCAB_SIZE,
    SHOGI_POSITION_STATE_FEATURE_ID,
    SIDE_TO_MOVE_BLACK_FEATURE_ID,
    SIDE_TO_MOVE_WHITE_FEATURE_ID,
)
from intrep.representation.inputs.shogi_position_features.position_square_features import (
    move_count_bucket_feature_id,
)


SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_SCHEMA_ID = "shogi_minimal_split_global_position_features"
SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT = 17
SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT = 81
SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_FIELD_COUNT = 1
SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT = (
    SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT + SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT
)
SHOGI_MINIMAL_SPLIT_GLOBAL_STATE_ELEMENT_INDEX = 0
SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_OFFSET = SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT
_SFEN_PIECE_TYPES = {
    "P": 1,
    "L": 2,
    "N": 3,
    "S": 4,
    "G": 5,
    "B": 6,
    "R": 7,
    "K": 8,
    "+P": 9,
    "+L": 10,
    "+N": 11,
    "+S": 12,
    "+B": 13,
    "+R": 14,
}
_HAND_PIECE_INDEX_BY_TYPE = {piece_type: index for index, piece_type in enumerate(HAND_PIECE_TYPES)}
_EMPTY_FEATURE_IDS = torch.empty((0, 0), dtype=torch.long)
_EMPTY_PAIR_RELATION_EDGES = ShogiPairRelationEdges(
    source_element_indices=torch.empty((0,), dtype=torch.long),
    target_element_indices=torch.empty((0,), dtype=torch.long),
    relation_ids=torch.empty((0,), dtype=torch.long),
)


def shogi_minimal_split_global_position_feature_manifest() -> dict[str, object]:
    return {
        "input_schema_id": SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_INPUT_SCHEMA_ID,
        "coordinate_system": "side_to_move_relative_180_rotation",
        "representation_element_count": SHOGI_MINIMAL_SPLIT_GLOBAL_ELEMENT_COUNT,
        "global_element_count": SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT,
        "square_element_count": SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT,
        "square_field_count": SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_FIELD_COUNT,
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


def shogi_minimal_split_global_position_feature_manifest_hash() -> str:
    payload = json.dumps(shogi_minimal_split_global_position_feature_manifest(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST = shogi_minimal_split_global_position_feature_manifest()
SHOGI_MINIMAL_SPLIT_GLOBAL_POSITION_FEATURE_MANIFEST_HASH = shogi_minimal_split_global_position_feature_manifest_hash()


def shogi_minimal_split_global_position_features_from_sfen(position_sfen: str) -> ShogiPositionFeatures:
    board_sfen, turn_sfen, hand_sfen, move_number_sfen = _split_minimal_sfen(position_sfen)
    own_is_black = turn_sfen == "b"
    global_feature_ids = torch.tensor(
        [
            SHOGI_POSITION_STATE_FEATURE_ID,
            SIDE_TO_MOVE_BLACK_FEATURE_ID if own_is_black else SIDE_TO_MOVE_WHITE_FEATURE_ID,
            move_count_bucket_feature_id(int(move_number_sfen)),
            *_hand_feature_ids_from_sfen(hand_sfen, own_is_black=own_is_black),
        ],
        dtype=torch.long,
    )
    return ShogiPositionFeatures(
        global_feature_ids=global_feature_ids,
        square_feature_ids=torch.tensor(
            _square_feature_id_rows_from_sfen(board_sfen, own_is_black=own_is_black),
            dtype=torch.long,
        ),
        piece_feature_ids=_EMPTY_FEATURE_IDS,
        line_feature_ids=_EMPTY_FEATURE_IDS,
        pair_relation_edges=_EMPTY_PAIR_RELATION_EDGES,
    )


def _split_minimal_sfen(position_sfen: str) -> tuple[str, str, str, str]:
    parts = position_sfen.split()
    if len(parts) < 4:
        raise ValueError("shogi SFEN must contain board, side to move, hands, and move number")
    board_sfen, turn_sfen, hand_sfen, move_number_sfen = parts[:4]
    if turn_sfen not in {"b", "w"}:
        raise ValueError(f"unsupported shogi SFEN side to move: {turn_sfen}")
    return board_sfen, turn_sfen, hand_sfen, move_number_sfen


def _square_feature_id_rows_from_sfen(board_sfen: str, *, own_is_black: bool) -> list[list[int]]:
    square_feature_ids = [EMPTY_SQUARE_FEATURE_ID] * SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT
    absolute_square = 0
    promoted = False
    for char in board_sfen:
        if char == "/":
            continue
        if char.isdigit():
            absolute_square += int(char)
            continue
        if char == "+":
            promoted = True
            continue
        piece_type_key = f"+{char.upper()}" if promoted else char.upper()
        promoted = False
        piece_type = _SFEN_PIECE_TYPES[piece_type_key]
        piece_is_black = char.isupper()
        relative_square = absolute_square if own_is_black else 80 - absolute_square
        offset = OWN_PIECE_OFFSET if piece_is_black == own_is_black else OPPONENT_PIECE_OFFSET
        square_feature_ids[relative_square] = offset + piece_type - 1
        absolute_square += 1
    if absolute_square != SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT:
        raise ValueError(f"shogi SFEN board must contain 81 squares: {board_sfen}")
    return [[feature_id] for feature_id in square_feature_ids]


def _hand_feature_ids_from_sfen(hand_sfen: str, *, own_is_black: bool) -> list[int]:
    black_hand_counts, white_hand_counts = _hand_counts_from_sfen(hand_sfen)
    own_hand_counts = black_hand_counts if own_is_black else white_hand_counts
    opponent_hand_counts = white_hand_counts if own_is_black else black_hand_counts
    return [
        *(OWN_HAND_OFFSET + min(count, HAND_COUNT_BUCKET_MAX) for count in own_hand_counts),
        *(OPPONENT_HAND_OFFSET + min(count, HAND_COUNT_BUCKET_MAX) for count in opponent_hand_counts),
    ]


def _hand_counts_from_sfen(hand_sfen: str) -> tuple[list[int], list[int]]:
    black_hand_counts = [0] * len(HAND_PIECE_TYPES)
    white_hand_counts = [0] * len(HAND_PIECE_TYPES)
    if hand_sfen == "-":
        return black_hand_counts, white_hand_counts
    count = 0
    for char in hand_sfen:
        if char.isdigit():
            count = count * 10 + int(char)
            continue
        piece_type = _SFEN_PIECE_TYPES[char.upper()]
        piece_count = count or 1
        count = 0
        hand_counts = black_hand_counts if char.isupper() else white_hand_counts
        hand_counts[_HAND_PIECE_INDEX_BY_TYPE[piece_type]] += piece_count
    return black_hand_counts, white_hand_counts


def validate_shogi_minimal_split_global_position_feature_structure(features: ShogiPositionFeatures) -> None:
    validate_integer_tensor_shape(
        "global_feature_ids",
        features.global_feature_ids,
        (SHOGI_MINIMAL_SPLIT_GLOBAL_GLOBAL_ELEMENT_COUNT,),
    )
    validate_integer_tensor_shape(
        "square_feature_ids",
        features.square_feature_ids,
        (SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_ELEMENT_COUNT, SHOGI_MINIMAL_SPLIT_GLOBAL_SQUARE_FIELD_COUNT),
    )
    validate_integer_tensor_shape("piece_feature_ids", features.piece_feature_ids, (0, 0))
    validate_integer_tensor_shape("line_feature_ids", features.line_feature_ids, (0, 0))
    validate_empty_pair_relation_edges(features.pair_relation_edges, context="minimal-split-global")
