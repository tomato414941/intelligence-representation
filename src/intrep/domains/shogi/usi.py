from __future__ import annotations

import shogi


SHOGI_USI_DROP_PIECE_TYPES_BY_CODE = {
    "P": shogi.PAWN,
    "L": shogi.LANCE,
    "N": shogi.KNIGHT,
    "S": shogi.SILVER,
    "G": shogi.GOLD,
    "B": shogi.BISHOP,
    "R": shogi.ROOK,
}
_RANK_INDEX_BY_CODE = {code: index for index, code in enumerate("abcdefghi")}


def shogi_usi_square_to_absolute_square(file_code: str, rank_code: str) -> int:
    file_number = ord(file_code) - ord("0")
    rank_index = _RANK_INDEX_BY_CODE.get(rank_code)
    if file_number < 1 or file_number > 9 or rank_index is None:
        raise ValueError(f"invalid shogi USI square: {file_code}{rank_code}")
    return rank_index * 9 + (9 - file_number)


def shogi_usi_move_parts(move_usi: str) -> tuple[int | None, int, bool, int | None]:
    if len(move_usi) == 4 and move_usi[1] == "*":
        drop_piece_type = SHOGI_USI_DROP_PIECE_TYPES_BY_CODE.get(move_usi[0])
        if drop_piece_type is None:
            raise ValueError(f"invalid shogi USI drop move: {move_usi}")
        return (
            None,
            shogi_usi_square_to_absolute_square(move_usi[2], move_usi[3]),
            False,
            drop_piece_type,
        )
    if len(move_usi) not in {4, 5}:
        raise ValueError(f"invalid shogi USI move: {move_usi}")
    promotion = len(move_usi) == 5 and move_usi[4] == "+"
    if len(move_usi) == 5 and not promotion:
        raise ValueError(f"invalid shogi USI promotion marker: {move_usi}")
    return (
        shogi_usi_square_to_absolute_square(move_usi[0], move_usi[1]),
        shogi_usi_square_to_absolute_square(move_usi[2], move_usi[3]),
        promotion,
        None,
    )
