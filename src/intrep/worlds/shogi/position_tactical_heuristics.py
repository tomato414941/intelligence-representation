from __future__ import annotations

import shogi

from intrep.worlds.shogi.position_geometry import opponent_color, relative_to_absolute_square
from intrep.worlds.shogi.position_line_features import line_kind_index, piece_slides_on_line, squares_for_line_index
from intrep.worlds.shogi.position_schema import *


def counterfactual_removal_token_id_rows(board: shogi.Board) -> list[list[int]]:
    rows: list[list[int]] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        piece = board.piece_at(absolute_square)
        if piece is None:
            rows.append(
                [
                    COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET,
                    COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET,
                    COUNTERFACTUAL_REMOVAL_SLIDER_BLOCKER_OFFSET,
                ]
            )
            continue
        removed_board = shogi.Board(board.sfen())
        removed_board.remove_piece_at(absolute_square)
        rows.append(
            [
                COUNTERFACTUAL_REMOVAL_SELF_CHECK_OFFSET
                + int(king_is_attacked(removed_board, board.turn)),
                COUNTERFACTUAL_REMOVAL_OPPONENT_CHECK_OFFSET
                + int(king_is_attacked(removed_board, opponent_color(board.turn))),
                COUNTERFACTUAL_REMOVAL_SLIDER_BLOCKER_OFFSET
                + int(slider_line_blocker(board, absolute_square)),
            ]
        )
    return rows


def gift_flow_token_id_rows(board: shogi.Board) -> list[list[int]]:
    own_king_zone = king_zone_absolute_squares(board, board.turn)
    opponent_king_zone = king_zone_absolute_squares(board, opponent_color(board.turn))
    rows: list[list[int]] = []
    for relative_square in range(BOARD_TOKEN_COUNT):
        absolute_square = relative_to_absolute_square(relative_square, board.turn)
        piece = board.piece_at(absolute_square)
        if piece is None:
            rows.append([GIFT_DANGER_OFFSET, CAPTURE_FLOW_OPPORTUNITY_OFFSET])
            continue
        hand_piece_type = hand_piece_type_after_capture(piece.piece_type)
        gift_danger = piece.color == board.turn and has_pseudo_drop_target(
            board,
            opponent_color(board.turn),
            hand_piece_type,
            own_king_zone,
        )
        capture_opportunity = piece.color == opponent_color(board.turn) and has_pseudo_drop_target(
            board,
            board.turn,
            hand_piece_type,
            opponent_king_zone,
        )
        rows.append(
            [
                GIFT_DANGER_OFFSET + int(gift_danger),
                CAPTURE_FLOW_OPPORTUNITY_OFFSET + int(capture_opportunity),
            ]
        )
    return rows


def king_is_attacked(board: shogi.Board, color: int) -> bool:
    king_square = board.king_squares[color]
    return king_square is not None and bool(board.attackers(opponent_color(color), int(king_square)))


def slider_line_blocker(board: shogi.Board, absolute_square: int) -> bool:
    for line_index in range(LINE_TOKEN_COUNT):
        relative_squares = squares_for_line_index(line_index)
        absolute_line = [relative_to_absolute_square(square, board.turn) for square in relative_squares]
        if absolute_square not in absolute_line:
            continue
        square_index = absolute_line.index(absolute_square)
        before = absolute_line[:square_index]
        after = absolute_line[square_index + 1 :]
        if line_side_has_slider(board, before, line_kind=line_kind_index(line_index)) and line_side_has_piece(board, after):
            return True
        if line_side_has_slider(board, after, line_kind=line_kind_index(line_index)) and line_side_has_piece(board, before):
            return True
    return False


def line_side_has_slider(board: shogi.Board, squares: list[int], *, line_kind: int) -> bool:
    for square in squares:
        piece = board.piece_at(square)
        if piece is not None and piece_slides_on_line(piece.piece_type, line_kind):
            return True
    return False


def line_side_has_piece(board: shogi.Board, squares: list[int]) -> bool:
    return any(board.piece_at(square) is not None for square in squares)


def king_zone_absolute_squares(board: shogi.Board, color: int) -> set[int]:
    king_square = board.king_squares[color]
    if king_square is None:
        return set()
    king_file = int(king_square) % 9
    king_rank = int(king_square) // 9
    zone: set[int] = set()
    for rank_delta in (-1, 0, 1):
        for file_delta in (-1, 0, 1):
            file_index = king_file + file_delta
            rank_index = king_rank + rank_delta
            if 0 <= file_index < 9 and 0 <= rank_index < 9:
                zone.add(rank_index * 9 + file_index)
    return zone


def has_pseudo_drop_target(board: shogi.Board, color: int, piece_type: int, candidate_squares: set[int]) -> bool:
    if piece_type not in HAND_PIECE_TYPES:
        return False
    for square in candidate_squares:
        if board.piece_at(square) is None and piece_type_can_drop_on_square(piece_type, color, square):
            return True
    return False


def piece_type_can_drop_on_square(piece_type: int, color: int, absolute_square: int) -> bool:
    rank = absolute_square // 9
    if piece_type in (shogi.PAWN, shogi.LANCE):
        return (color == shogi.BLACK and rank != 0) or (color == shogi.WHITE and rank != 8)
    if piece_type == shogi.KNIGHT:
        return (color == shogi.BLACK and rank > 1) or (color == shogi.WHITE and rank < 7)
    return True


def hand_piece_type_after_capture(piece_type: int) -> int:
    for base_piece_type, promoted_piece_type in enumerate(shogi.PIECE_PROMOTED):
        if promoted_piece_type == piece_type:
            return base_piece_type
    return piece_type
