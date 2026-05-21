from __future__ import annotations

import torch

from intrep.domains.shogi.coordinates import SHOGI_SQUARE_COUNT

SHOGI_SQUARE_RELATIVE_POSITION_BUCKET_COUNT = 17 * 17


def shogi_square_relative_position_relation_ids() -> torch.Tensor:
    square_count = SHOGI_SQUARE_COUNT
    relation_ids = torch.empty((square_count, square_count), dtype=torch.long)
    for from_square in range(square_count):
        from_file = from_square % 9
        from_rank = from_square // 9
        for to_square in range(square_count):
            to_file = to_square % 9
            to_rank = to_square // 9
            file_delta = to_file - from_file
            rank_delta = to_rank - from_rank
            relation_ids[from_square, to_square] = (rank_delta + 8) * 17 + file_delta + 8
    return relation_ids
