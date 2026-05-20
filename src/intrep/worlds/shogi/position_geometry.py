from __future__ import annotations

import shogi

from intrep.worlds.shogi.position_schema import BOARD_TOKEN_COUNT


def king_relative_offset_bucket(relative_square: int, relative_king_square: int) -> int:
    square_file = relative_square % 9
    square_rank = relative_square // 9
    king_file = relative_king_square % 9
    king_rank = relative_king_square // 9
    file_delta = square_file - king_file
    rank_delta = square_rank - king_rank
    return (rank_delta + 8) * 17 + file_delta + 8


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
