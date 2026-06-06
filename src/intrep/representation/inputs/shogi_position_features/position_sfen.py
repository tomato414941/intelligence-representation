from __future__ import annotations

import torch

from intrep.representation.inputs.shogi_position_features.position_features import ShogiPairRelationEdges
from intrep.representation.inputs.shogi_position_features.position_schema import (
    EMPTY_SQUARE_FEATURE_ID,
    HAND_COUNT_BUCKET_MAX,
    HAND_PIECE_TYPES,
    OPPONENT_HAND_OFFSET,
    OPPONENT_PIECE_OFFSET,
    OWN_HAND_OFFSET,
    OWN_PIECE_OFFSET,
    SQUARE_ATTACK_PIECE_TYPE_COUNT,
)


SHOGI_SFEN_POSITION_SQUARE_COUNT = 81
SHOGI_EMPTY_FEATURE_IDS = torch.empty((0, 0), dtype=torch.long)
SHOGI_EMPTY_PAIR_RELATION_EDGES = ShogiPairRelationEdges(
    source_element_indices=torch.empty((0,), dtype=torch.long),
    target_element_indices=torch.empty((0,), dtype=torch.long),
    relation_ids=torch.empty((0,), dtype=torch.long),
)
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


def split_shogi_sfen(position_sfen: str) -> tuple[str, str, str, str]:
    parts = position_sfen.split()
    if len(parts) < 4:
        raise ValueError("shogi SFEN must contain board, side to move, hands, and move number")
    board_sfen, turn_sfen, hand_sfen, move_number_sfen = parts[:4]
    if turn_sfen not in {"b", "w"}:
        raise ValueError(f"unsupported shogi SFEN side to move: {turn_sfen}")
    return board_sfen, turn_sfen, hand_sfen, move_number_sfen


def shogi_square_piece_feature_id_rows_from_sfen(board_sfen: str, *, own_is_black: bool) -> list[list[int]]:
    square_feature_ids = [EMPTY_SQUARE_FEATURE_ID] * SHOGI_SFEN_POSITION_SQUARE_COUNT
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
    if absolute_square != SHOGI_SFEN_POSITION_SQUARE_COUNT:
        raise ValueError(f"shogi SFEN board must contain 81 squares: {board_sfen}")
    return [[feature_id] for feature_id in square_feature_ids]


def shogi_current_square_piece_plane_feature_id_rows_from_sfen(
    board_sfen: str,
    *,
    own_is_black: bool,
) -> list[list[int]]:
    rows = [[EMPTY_SQUARE_FEATURE_ID] * (SQUARE_ATTACK_PIECE_TYPE_COUNT * 2) for _ in range(SHOGI_SFEN_POSITION_SQUARE_COUNT)]
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
        if piece_is_black == own_is_black:
            rows[relative_square][piece_type - 1] = OWN_PIECE_OFFSET + piece_type - 1
        else:
            rows[relative_square][SQUARE_ATTACK_PIECE_TYPE_COUNT + piece_type - 1] = OPPONENT_PIECE_OFFSET + piece_type - 1
        absolute_square += 1
    if absolute_square != SHOGI_SFEN_POSITION_SQUARE_COUNT:
        raise ValueError(f"shogi SFEN board must contain 81 squares: {board_sfen}")
    return rows


def shogi_hand_feature_ids_from_sfen(hand_sfen: str, *, own_is_black: bool) -> list[int]:
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
