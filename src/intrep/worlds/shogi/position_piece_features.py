from __future__ import annotations

import shogi

from intrep.worlds.shogi.position_derived import PieceSlotRelationInfo, _ShogiPositionDerivedRelations
from intrep.worlds.shogi.position_geometry import opponent_color, relative_to_absolute_square
from intrep.worlds.shogi.position_schema import *
from intrep.worlds.shogi.position_square_features import king_relative_square_feature_id, piece_feature_id


def piece_feature_ids(board: shogi.Board) -> list[int]:
    return [feature_id for row in piece_feature_id_rows(board) for feature_id in row]


def piece_feature_id_rows(
    board: shogi.Board,
    *,
    derived: _ShogiPositionDerivedRelations | None = None,
) -> list[list[int]]:
    piece_features: list[int] = []
    if derived is None:
        from intrep.worlds.shogi.position_building import _shogi_position_derived_relations

        derived = _shogi_position_derived_relations(board)
    for relative_square in range(SQUARE_ELEMENT_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        piece = board.piece_at(absolute_square)
        if piece is not None:
            piece_features.extend(
                board_piece_slot_feature_ids(
                    board,
                    piece,
                    relative_square,
                    counterfactual_features=derived.counterfactual_removal_feature_rows[relative_square],
                    drop_potential_features=derived.drop_potential_feature_rows[relative_square],
                )
            )
    piece_features.extend(hand_piece_slot_feature_ids(board))
    empty_slot_count = PIECE_SLOT_COUNT - len(piece_features) // PIECE_FEATURE_COUNT
    if empty_slot_count < 0:
        raise ValueError("shogi board contains more pieces than supported piece slots")
    for _ in range(empty_slot_count):
        piece_features.extend(empty_piece_slot_feature_ids())
    return [
        piece_features[index : index + PIECE_FEATURE_COUNT]
        for index in range(0, len(piece_features), PIECE_FEATURE_COUNT)
    ]


def piece_slot_relation_infos(board: shogi.Board) -> list[PieceSlotRelationInfo]:
    infos: list[PieceSlotRelationInfo] = []
    for relative_square in range(SQUARE_ELEMENT_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        piece = board.piece_at(absolute_square)
        if piece is not None:
            infos.append(PieceSlotRelationInfo(piece=piece, location_kind="board", relative_square=relative_square))
    for color in (board.turn, opponent_color(board.turn)):
        for piece_type in HAND_PIECE_TYPES:
            for _ in range(board.pieces_in_hand[color][piece_type]):
                infos.append(
                    PieceSlotRelationInfo(
                        piece=shogi.Piece(piece_type, color),
                        location_kind="hand",
                        relative_square=None,
                    )
                )
    while len(infos) < PIECE_SLOT_COUNT:
        infos.append(PieceSlotRelationInfo(piece=None, location_kind="empty", relative_square=None))
    if len(infos) > PIECE_SLOT_COUNT:
        raise ValueError("shogi board contains more pieces than supported piece slots")
    return infos


def board_piece_slot_feature_ids(
    board: shogi.Board,
    piece: shogi.Piece,
    relative_square: int,
    *,
    counterfactual_features: list[int],
    drop_potential_features: list[int],
) -> list[int]:
    return [
        PIECE_LOCATION_BOARD_FEATURE_ID,
        piece_feature_id(piece, own_color=board.turn),
        PIECE_SQUARE_OFFSET + relative_square,
        king_relative_square_feature_id(
            board,
            board.turn,
            relative_square,
            offset=OWN_KING_RELATIVE_SQUARE_OFFSET,
        ),
        king_relative_square_feature_id(
            board,
            opponent_color(board.turn),
            relative_square,
            offset=OPPONENT_KING_RELATIVE_SQUARE_OFFSET,
        ),
        *counterfactual_features,
        *drop_potential_features,
    ]


def hand_piece_slot_feature_ids(board: shogi.Board) -> list[int]:
    feature_ids: list[int] = []
    for color in (board.turn, opponent_color(board.turn)):
        for piece_type in HAND_PIECE_TYPES:
            for _ in range(board.pieces_in_hand[color][piece_type]):
                feature_ids.extend(
                    hand_piece_feature_ids(
                        shogi.Piece(piece_type, color),
                        own_color=board.turn,
                    )
                )
    return feature_ids


def hand_piece_feature_ids(piece: shogi.Piece, *, own_color: int) -> list[int]:
    return [
        PIECE_LOCATION_HAND_FEATURE_ID,
        piece_feature_id(piece, own_color=own_color),
        PIECE_SQUARE_UNKNOWN_FEATURE_ID,
        OWN_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        OPPONENT_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET,
        COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET,
        COUNTERFACTUAL_REMOVAL_COARSE_SLIDER_BLOCKER_OFFSET,
        OPPONENT_DROP_POTENTIAL_AFTER_LOSING_PIECE_OFFSET,
        OWN_DROP_POTENTIAL_AFTER_CAPTURING_PIECE_OFFSET,
    ]


def empty_piece_slot_feature_ids() -> list[int]:
    return [
        PIECE_LOCATION_EMPTY_FEATURE_ID,
        EMPTY_SQUARE_FEATURE_ID,
        PIECE_SQUARE_UNKNOWN_FEATURE_ID,
        OWN_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        OPPONENT_KING_RELATIVE_SQUARE_OFFSET + KING_RELATIVE_SQUARE_BUCKET_UNKNOWN,
        COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET,
        COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET,
        COUNTERFACTUAL_REMOVAL_COARSE_SLIDER_BLOCKER_OFFSET,
        OPPONENT_DROP_POTENTIAL_AFTER_LOSING_PIECE_OFFSET,
        OWN_DROP_POTENTIAL_AFTER_CAPTURING_PIECE_OFFSET,
    ]
