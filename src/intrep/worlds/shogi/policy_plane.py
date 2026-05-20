from __future__ import annotations

import shogi
import torch

from intrep.worlds.shogi.position_encoding import (
    SQUARE_ELEMENT_COUNT,
    absolute_to_relative_square,
    relative_to_absolute_square,
)


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

SHORT_MOVE_TYPE_OFFSET = 0
SHORT_PROMOTE_MOVE_TYPE_OFFSET = SHORT_MOVE_TYPE_OFFSET + len(DIRECTION_DELTAS)
LONG_MOVE_TYPE_OFFSET = SHORT_PROMOTE_MOVE_TYPE_OFFSET + len(DIRECTION_DELTAS)
LONG_PROMOTE_MOVE_TYPE_OFFSET = LONG_MOVE_TYPE_OFFSET + len(DIRECTION_DELTAS)
KNIGHT_MOVE_TYPE_OFFSET = LONG_PROMOTE_MOVE_TYPE_OFFSET + len(DIRECTION_DELTAS)
KNIGHT_PROMOTE_MOVE_TYPE_OFFSET = KNIGHT_MOVE_TYPE_OFFSET + len(KNIGHT_DELTAS)
DROP_MOVE_TYPE_OFFSET = KNIGHT_PROMOTE_MOVE_TYPE_OFFSET + len(KNIGHT_DELTAS)
SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT = DROP_MOVE_TYPE_OFFSET + len(DROP_PIECE_TYPES)
SHOGI_POLICY_PLANE_ACTION_COUNT = SQUARE_ELEMENT_COUNT * SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT


def shogi_policy_plane_action_index(move_usi: str, *, turn: int) -> int:
    move = shogi.Move.from_usi(move_usi)
    to_square = absolute_to_relative_square(int(move.to_square), turn)
    move_type = shogi_policy_plane_move_type(move, turn=turn)
    return to_square * SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT + move_type


def shogi_policy_plane_move_type(move: shogi.Move, *, turn: int) -> int:
    if move.drop_piece_type is not None:
        return DROP_MOVE_TYPE_OFFSET + DROP_PIECE_TYPES.index(int(move.drop_piece_type))
    if move.from_square is None:
        raise ValueError("non-drop shogi move must have from_square")
    from_square = absolute_to_relative_square(int(move.from_square), turn)
    to_square = absolute_to_relative_square(int(move.to_square), turn)
    delta = to_square - from_square
    if delta in KNIGHT_DELTAS:
        offset = KNIGHT_PROMOTE_MOVE_TYPE_OFFSET if move.promotion else KNIGHT_MOVE_TYPE_OFFSET
        return offset + KNIGHT_DELTAS.index(delta)
    direction = _direction_delta(delta)
    if direction is None:
        raise ValueError(f"unsupported shogi policy-plane move geometry: {move.usi()}")
    offset = _direction_offset(delta, promoted=bool(move.promotion))
    return offset + DIRECTION_DELTAS.index(direction)


def shogi_policy_plane_legal_mask(board: shogi.Board) -> torch.Tensor:
    mask = torch.zeros(SHOGI_POLICY_PLANE_ACTION_COUNT, dtype=torch.bool)
    for move in board.legal_moves:
        mask[shogi_policy_plane_action_index(move.usi(), turn=board.turn)] = True
    return mask


def shogi_policy_plane_legal_move_by_action_index(board: shogi.Board) -> dict[int, str]:
    return {
        shogi_policy_plane_action_index(move.usi(), turn=board.turn): move.usi()
        for move in board.legal_moves
    }


def shogi_policy_plane_move_from_action_index(action_index: int, board: shogi.Board) -> str:
    legal_moves_by_index = shogi_policy_plane_legal_move_by_action_index(board)
    try:
        return legal_moves_by_index[action_index]
    except KeyError as exc:
        raise ValueError("action_index does not correspond to a legal move in this board") from exc


def _direction_delta(delta: int) -> int | None:
    for direction in DIRECTION_DELTAS:
        if delta % direction == 0:
            distance = delta // direction
            if distance > 0:
                return direction
    return None


def _direction_offset(delta: int, *, promoted: bool) -> int:
    if abs(delta) in {1, 8, 9, 10}:
        return SHORT_PROMOTE_MOVE_TYPE_OFFSET if promoted else SHORT_MOVE_TYPE_OFFSET
    return LONG_PROMOTE_MOVE_TYPE_OFFSET if promoted else LONG_MOVE_TYPE_OFFSET


def shogi_policy_plane_to_square(action_index: int, *, turn: int) -> int:
    relative_to_square = action_index // SHOGI_POLICY_PLANE_MOVE_TYPE_COUNT
    return relative_to_absolute_square(relative_to_square, turn)
