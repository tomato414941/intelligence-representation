from __future__ import annotations

import shogi
import torch

from intrep.domains.shogi.coordinates import (
    SHOGI_SQUARE_COUNT,
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

MOVE_TYPE_DELTAS = DIRECTION_DELTAS + KNIGHT_DELTAS
MOVE_TYPE_OFFSET = 0
PROMOTE_MOVE_TYPE_OFFSET = MOVE_TYPE_OFFSET + len(MOVE_TYPE_DELTAS)
DROP_MOVE_TYPE_OFFSET = PROMOTE_MOVE_TYPE_OFFSET + len(MOVE_TYPE_DELTAS)
SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT = DROP_MOVE_TYPE_OFFSET + len(DROP_PIECE_TYPES)
SHOGI_ACTION_PLANE_POLICY_ACTION_COUNT = SHOGI_SQUARE_COUNT * SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT


def shogi_action_plane_policy_action_index(move_usi: str, *, turn: int) -> int:
    move = shogi.Move.from_usi(move_usi)
    to_square = absolute_to_relative_square(int(move.to_square), turn)
    move_type = shogi_action_plane_policy_move_type(move, turn=turn)
    return to_square * SHOGI_ACTION_PLANE_POLICY_MOVE_TYPE_COUNT + move_type


def shogi_action_plane_policy_move_type(move: shogi.Move, *, turn: int) -> int:
    if move.drop_piece_type is not None:
        return DROP_MOVE_TYPE_OFFSET + DROP_PIECE_TYPES.index(int(move.drop_piece_type))
    if move.from_square is None:
        raise ValueError("non-drop shogi move must have from_square")
    from_square = absolute_to_relative_square(int(move.from_square), turn)
    to_square = absolute_to_relative_square(int(move.to_square), turn)
    delta = to_square - from_square
    if delta in KNIGHT_DELTAS:
        move_type_delta = delta
    else:
        move_type_delta = _direction_delta(from_square, to_square)
        if move_type_delta is None:
            raise ValueError(f"unsupported shogi action-plane policy move geometry: {move.usi()}")
    offset = PROMOTE_MOVE_TYPE_OFFSET if move.promotion else MOVE_TYPE_OFFSET
    return offset + MOVE_TYPE_DELTAS.index(move_type_delta)


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
