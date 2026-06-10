from __future__ import annotations

import shogi

from intrep.worlds.shogi.coordinates import opponent_color, relative_to_absolute_square
from intrep.representation.inputs.shogi_position_features.position_schema import *


def line_feature_ids(board: shogi.Board) -> list[int]:
    feature_ids: list[int] = []
    for row in line_feature_id_rows(board):
        feature_ids.extend(row)
    return feature_ids


def line_feature_id_rows(board: shogi.Board) -> list[list[int]]:
    return [line_element_feature_ids(board, line_index) for line_index in range(LINE_ELEMENT_COUNT)]


def line_element_feature_ids(board: shogi.Board, line_index: int) -> list[int]:
    line_kind = line_kind_index(line_index)
    relative_squares = squares_for_line_index(line_index)
    absolute_squares = {relative_to_absolute_square(square, board.turn) for square in relative_squares}
    own_king_on_line = king_on_absolute_squares(board, board.turn, absolute_squares)
    opponent_king_on_line = king_on_absolute_squares(board, opponent_color(board.turn), absolute_squares)
    own_slider_on_line = slider_on_absolute_squares(board, board.turn, absolute_squares, line_kind=line_kind)
    opponent_slider_on_line = slider_on_absolute_squares(
        board,
        opponent_color(board.turn),
        absolute_squares,
        line_kind=line_kind,
    )
    occupancy_count = sum(1 for square in absolute_squares if board.piece_at(square) is not None)
    return [
        LINE_KIND_OFFSET + line_kind,
        LINE_OWN_KING_ON_LINE_OFFSET + int(own_king_on_line),
        LINE_OPPONENT_KING_ON_LINE_OFFSET + int(opponent_king_on_line),
        LINE_OWN_SLIDER_ON_LINE_OFFSET + int(own_slider_on_line),
        LINE_OPPONENT_SLIDER_ON_LINE_OFFSET + int(opponent_slider_on_line),
        LINE_OCCUPANCY_COUNT_OFFSET + min(occupancy_count, LINE_OCCUPANCY_COUNT_MAX),
    ]


def line_kind_index(line_index: int) -> int:
    if line_index < 9:
        return 0
    if line_index < 18:
        return 1
    if line_index < 35:
        return 2
    return 3


def squares_for_line_index(line_index: int) -> tuple[int, ...]:
    if line_index < 9:
        file_index = line_index
        return tuple(rank * 9 + file_index for rank in range(9))
    if line_index < 18:
        rank = line_index - 9
        return tuple(rank * 9 + file_index for file_index in range(9))
    if line_index < 35:
        diagonal = line_index - 18
        return tuple(square for square in range(SQUARE_ELEMENT_COUNT) if square // 9 + square % 9 == diagonal)
    diagonal = line_index - 35
    return tuple(square for square in range(SQUARE_ELEMENT_COUNT) if square // 9 - square % 9 + 8 == diagonal)


def king_on_absolute_squares(board: shogi.Board, color: int, absolute_squares: set[int]) -> bool:
    king_square = board.king_squares[color]
    return king_square is not None and int(king_square) in absolute_squares


def slider_on_absolute_squares(
    board: shogi.Board,
    color: int,
    absolute_squares: set[int],
    *,
    line_kind: int,
) -> bool:
    for square in absolute_squares:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color and piece_slides_on_line(piece.piece_type, line_kind):
            return True
    return False


def piece_slides_on_line(piece_type: int, line_kind: int) -> bool:
    if line_kind == 0:
        return piece_type in (shogi.LANCE, shogi.ROOK, shogi.PROM_ROOK)
    if line_kind == 1:
        return piece_type in (shogi.ROOK, shogi.PROM_ROOK)
    return piece_type in (shogi.BISHOP, shogi.PROM_BISHOP)
