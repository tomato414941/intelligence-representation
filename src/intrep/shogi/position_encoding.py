from __future__ import annotations

import shogi
import torch


BOARD_TOKEN_COUNT = 81
HAND_TOKEN_COUNT = 14
SHOGI_POSITION_TOKEN_COUNT = 1 + BOARD_TOKEN_COUNT + HAND_TOKEN_COUNT

EMPTY_SQUARE_TOKEN_ID = 0
BLACK_PIECE_OFFSET = 1
WHITE_PIECE_OFFSET = 15
SIDE_TO_MOVE_BLACK_TOKEN_ID = 29
SIDE_TO_MOVE_WHITE_TOKEN_ID = 30
HAND_BLACK_OFFSET = 31
HAND_WHITE_OFFSET = 38
SHOGI_POSITION_VOCAB_SIZE = 45

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
    token_ids.extend(square_token_id(board.piece_at(square)) for square in range(BOARD_TOKEN_COUNT))
    token_ids.extend(hand_token_ids(board))
    return torch.tensor(token_ids, dtype=torch.long)


def side_to_move_token_id(color: int) -> int:
    if color == shogi.BLACK:
        return SIDE_TO_MOVE_BLACK_TOKEN_ID
    if color == shogi.WHITE:
        return SIDE_TO_MOVE_WHITE_TOKEN_ID
    raise ValueError(f"unsupported shogi color: {color}")


def square_token_id(piece: shogi.Piece | None) -> int:
    if piece is None:
        return EMPTY_SQUARE_TOKEN_ID
    if piece.color == shogi.BLACK:
        return BLACK_PIECE_OFFSET + int(piece.piece_type) - 1
    if piece.color == shogi.WHITE:
        return WHITE_PIECE_OFFSET + int(piece.piece_type) - 1
    raise ValueError(f"unsupported shogi piece color: {piece.color}")


def hand_token_ids(board: shogi.Board) -> list[int]:
    token_ids: list[int] = []
    for color, offset in ((shogi.BLACK, HAND_BLACK_OFFSET), (shogi.WHITE, HAND_WHITE_OFFSET)):
        pieces_in_hand = board.pieces_in_hand[color]
        for piece_type in HAND_PIECE_TYPES:
            count = pieces_in_hand[piece_type]
            token_ids.append(offset + min(count, 6))
    return token_ids
