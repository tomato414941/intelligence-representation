from __future__ import annotations

import shogi
import torch

from intrep.domains.shogi.coordinates import SHOGI_SQUARE_COUNT, absolute_to_relative_square


SHOGI_MOVE_FIELD_COUNT = 4
NO_FROM_SQUARE_ID = SHOGI_SQUARE_COUNT
NO_DROP_PIECE_ID = 0


def shogi_move_feature_ids(move_usi: str, *, turn: int) -> torch.Tensor:
    move = shogi.Move.from_usi(move_usi)
    from_square_id = (
        NO_FROM_SQUARE_ID
        if move.from_square is None
        else absolute_to_relative_square(int(move.from_square), turn)
    )
    to_square_id = absolute_to_relative_square(int(move.to_square), turn)
    promotion_id = int(move.promotion)
    drop_piece_id = NO_DROP_PIECE_ID if move.drop_piece_type is None else int(move.drop_piece_type)
    return torch.tensor(
        [from_square_id, to_square_id, promotion_id, drop_piece_id],
        dtype=torch.long,
    )


def shogi_legal_move_feature_ids(
    move_usis: tuple[str, ...],
    *,
    turn: int,
    max_legal_move_count: int,
) -> torch.Tensor:
    feature_ids = torch.zeros((max_legal_move_count, SHOGI_MOVE_FIELD_COUNT), dtype=torch.long)
    for index, move_usi in enumerate(move_usis):
        feature_ids[index] = shogi_move_feature_ids(move_usi, turn=turn)
    return feature_ids
