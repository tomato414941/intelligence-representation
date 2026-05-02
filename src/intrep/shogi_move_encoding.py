from __future__ import annotations

import shogi
import torch


SHOGI_MOVE_FEATURE_COUNT = 4
NO_FROM_SQUARE_ID = 81
NO_DROP_PIECE_ID = 0


def shogi_move_feature_ids(move_usi: str) -> torch.Tensor:
    move = shogi.Move.from_usi(move_usi)
    from_square_id = NO_FROM_SQUARE_ID if move.from_square is None else int(move.from_square)
    to_square_id = int(move.to_square)
    promotion_id = int(move.promotion)
    drop_piece_id = NO_DROP_PIECE_ID if move.drop_piece_type is None else int(move.drop_piece_type)
    return torch.tensor(
        [from_square_id, to_square_id, promotion_id, drop_piece_id],
        dtype=torch.long,
    )


def shogi_candidate_move_features(move_usis: tuple[str, ...], *, max_choice_count: int) -> torch.Tensor:
    features = torch.zeros((max_choice_count, SHOGI_MOVE_FEATURE_COUNT), dtype=torch.long)
    for index, move_usi in enumerate(move_usis):
        features[index] = shogi_move_feature_ids(move_usi)
    return features
