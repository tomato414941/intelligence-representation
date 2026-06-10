from __future__ import annotations

import shogi
import torch

from intrep.worlds.shogi.coordinates import (
    SHOGI_SQUARE_COUNT,
    absolute_to_relative_square,
    relative_to_absolute_square,
)
from intrep.worlds.shogi.usi import SHOGI_USI_DROP_PIECE_TYPES_BY_CODE, shogi_usi_square_to_absolute_square


DIRECTION_DELTAS = (
    -10,
    -9,
    -8,
    -1,
    1,
    8,
    9,
    10,
)
KNIGHT_DELTAS = (-19, -17)
DROP_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)

MOVE_TYPE_DELTAS = DIRECTION_DELTAS + KNIGHT_DELTAS
MOVE_TYPE_OFFSET_BY_DELTA = {delta: index for index, delta in enumerate(MOVE_TYPE_DELTAS)}
MOVE_TYPE_OFFSET = 0
PROMOTE_MOVE_TYPE_OFFSET = MOVE_TYPE_OFFSET + len(MOVE_TYPE_DELTAS)
DROP_MOVE_TYPE_OFFSET = PROMOTE_MOVE_TYPE_OFFSET + len(MOVE_TYPE_DELTAS)
DROP_MOVE_TYPE_OFFSET_BY_PIECE_TYPE = {
    piece_type: DROP_MOVE_TYPE_OFFSET + index for index, piece_type in enumerate(DROP_PIECE_TYPES)
}
DROP_MOVE_TYPE_OFFSET_BY_USI_CODE = {
    usi_code: DROP_MOVE_TYPE_OFFSET_BY_PIECE_TYPE[piece_type]
    for usi_code, piece_type in SHOGI_USI_DROP_PIECE_TYPES_BY_CODE.items()
}
SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT = DROP_MOVE_TYPE_OFFSET + len(DROP_PIECE_TYPES)
SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT = SHOGI_SQUARE_COUNT * SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT


def shogi_action_plane_policy_action_index(move_usi: str, *, turn: int) -> int:
    if len(move_usi) == 4 and move_usi[1] == "*":
        to_square = shogi_usi_square_to_absolute_square(move_usi[2], move_usi[3])
        relative_to_square = absolute_to_relative_square(to_square, turn)
        return relative_to_square * SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT + DROP_MOVE_TYPE_OFFSET_BY_USI_CODE[
            move_usi[0]
        ]
    if len(move_usi) not in {4, 5} or (len(move_usi) == 5 and move_usi[4] != "+"):
        raise ValueError(f"invalid shogi USI move: {move_usi}")
    from_square = shogi_usi_square_to_absolute_square(move_usi[0], move_usi[1])
    to_square = shogi_usi_square_to_absolute_square(move_usi[2], move_usi[3])
    relative_to_square = absolute_to_relative_square(to_square, turn)
    move_type = shogi_action_plane_policy_move_type(
        from_square=from_square,
        to_square=to_square,
        promotion=len(move_usi) == 5 and move_usi[4] == "+",
        drop_piece_type=None,
        turn=turn,
        move_usi=move_usi,
    )
    return relative_to_square * SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT + move_type


def shogi_action_plane_policy_move_type(
    *,
    from_square: int | None,
    to_square: int,
    promotion: bool,
    drop_piece_type: int | None,
    turn: int,
    move_usi: str,
) -> int:
    if drop_piece_type is not None:
        return DROP_MOVE_TYPE_OFFSET_BY_PIECE_TYPE[drop_piece_type]
    if from_square is None:
        raise ValueError("non-drop shogi move must have from_square")
    relative_from_square = absolute_to_relative_square(from_square, turn)
    relative_to_square = absolute_to_relative_square(to_square, turn)
    delta = relative_to_square - relative_from_square
    if delta in KNIGHT_DELTAS:
        move_type_delta = delta
    else:
        move_type_delta = _direction_delta(relative_from_square, relative_to_square)
        if move_type_delta is None:
            raise ValueError(f"unsupported shogi action-plane policy move geometry: {move_usi}")
    offset = PROMOTE_MOVE_TYPE_OFFSET if promotion else MOVE_TYPE_OFFSET
    return offset + MOVE_TYPE_OFFSET_BY_DELTA[move_type_delta]


def shogi_action_plane_policy_legal_mask(board: shogi.Board) -> torch.Tensor:
    mask = torch.zeros(SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT, dtype=torch.bool)
    for move in board.legal_moves:
        mask[shogi_action_plane_policy_action_index(move.usi(), turn=board.turn)] = True
    return mask


def shogi_action_plane_policy_legal_move_by_action_index(board: shogi.Board) -> dict[int, str]:
    legal_moves_by_index: dict[int, str] = {}
    for move in board.legal_moves:
        action_index = shogi_action_plane_policy_action_index(move.usi(), turn=board.turn)
        if action_index in legal_moves_by_index:
            raise ValueError(
                "duplicate shogi action-plane policy action index for legal moves: "
                f"{legal_moves_by_index[action_index]} and {move.usi()}"
            )
        legal_moves_by_index[action_index] = move.usi()
    return legal_moves_by_index


def shogi_action_plane_policy_move_from_action_index(action_index: int, board: shogi.Board) -> str:
    legal_moves_by_index = shogi_action_plane_policy_legal_move_by_action_index(board)
    try:
        return legal_moves_by_index[action_index]
    except KeyError as exc:
        raise ValueError("action_index does not correspond to a legal move in this board") from exc


def _direction_delta(from_square: int, to_square: int) -> int | None:
    from_file = from_square % 9
    from_rank = from_square // 9
    to_file = to_square % 9
    to_rank = to_square // 9
    file_delta = to_file - from_file
    rank_delta = to_rank - from_rank
    if file_delta == 0 and rank_delta != 0:
        return 9 if rank_delta > 0 else -9
    if rank_delta == 0 and file_delta != 0:
        return 1 if file_delta > 0 else -1
    if abs(file_delta) == abs(rank_delta) and file_delta != 0:
        file_step = 1 if file_delta > 0 else -1
        rank_step = 1 if rank_delta > 0 else -1
        return rank_step * 9 + file_step
    return None


def shogi_action_plane_policy_to_square(action_index: int, *, turn: int) -> int:
    relative_to_square = action_index // SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT
    return relative_to_absolute_square(relative_to_square, turn)
