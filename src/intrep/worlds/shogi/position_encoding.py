from __future__ import annotations

import shogi
import torch


BOARD_TOKEN_COUNT = 81
HAND_TOKEN_COUNT = 14
SHOGI_POSITION_TOKEN_COUNT = 1 + BOARD_TOKEN_COUNT + HAND_TOKEN_COUNT
SHOGI_POSITION_INPUT_ENCODING = "shogi_side_to_move_relative_v1"

EMPTY_SQUARE_TOKEN_ID = 0
OWN_PIECE_OFFSET = 1
OPPONENT_PIECE_OFFSET = 15
SIDE_TO_MOVE_BLACK_TOKEN_ID = 29
SIDE_TO_MOVE_WHITE_TOKEN_ID = 30
HAND_COUNT_TOKEN_MAX = 18
OWN_HAND_OFFSET = 31
OPPONENT_HAND_OFFSET = OWN_HAND_OFFSET + HAND_COUNT_TOKEN_MAX + 1
SHOGI_POSITION_VOCAB_SIZE = OPPONENT_HAND_OFFSET + HAND_COUNT_TOKEN_MAX + 1

HAND_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)


def shogi_position_token_ids_from_sfen(position_sfen: str) -> torch.Tensor:
    board = shogi.Board(position_sfen)
    token_ids = [side_to_move_token_id(board.turn)]
    token_ids.extend(relative_square_token_id(board, square) for square in range(BOARD_TOKEN_COUNT))
    token_ids.extend(hand_token_ids(board))
    return torch.tensor(token_ids, dtype=torch.long)


def side_to_move_token_id(color: int) -> int:
    if color == shogi.BLACK:
        return SIDE_TO_MOVE_BLACK_TOKEN_ID
    if color == shogi.WHITE:
        return SIDE_TO_MOVE_WHITE_TOKEN_ID
    raise ValueError(f"unsupported shogi color: {color}")


def relative_square_token_id(board: shogi.Board, relative_square: int) -> int:
    absolute_square = relative_to_absolute_square(relative_square, board.turn)
    return piece_token_id(board.piece_at(absolute_square), own_color=board.turn)


def piece_token_id(piece: shogi.Piece | None, *, own_color: int) -> int:
    if piece is None:
        return EMPTY_SQUARE_TOKEN_ID
    if piece.color == own_color:
        return OWN_PIECE_OFFSET + int(piece.piece_type) - 1
    if piece.color == opponent_color(own_color):
        return OPPONENT_PIECE_OFFSET + int(piece.piece_type) - 1
    raise ValueError(f"unsupported shogi piece color: {piece.color}")


def hand_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for color, offset in ((board.turn, OWN_HAND_OFFSET), (opponent_color(board.turn), OPPONENT_HAND_OFFSET)):
        pieces_in_hand = board.pieces_in_hand[color]
        for piece_type in HAND_PIECE_TYPES:
            count = pieces_in_hand[piece_type]
            token_ids.append(offset + min(count, HAND_COUNT_TOKEN_MAX))
    return token_ids


def absolute_to_relative_square(square: int, turn: int) -> int:
    if turn == shogi.BLACK:
        return square
    if turn == shogi.WHITE:
        return BOARD_TOKEN_COUNT - 1 - square
    raise ValueError(f"unsupported shogi color: {turn}")


def relative_to_absolute_square(square: int, turn: int) -> int:
    return absolute_to_relative_square(square, turn)


def opponent_color(color: int) -> int:
    if color == shogi.BLACK:
        return shogi.WHITE
    if color == shogi.WHITE:
        return shogi.BLACK
    raise ValueError(f"unsupported shogi color: {color}")
