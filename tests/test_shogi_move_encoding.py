import unittest

import shogi
import torch

from intrep.representation.outputs.shogi_legal_move_encoding import (
    NO_DROP_PIECE_ID,
    NO_FROM_SQUARE_ID,
    SHOGI_MOVE_FIELD_COUNT,
    shogi_legal_move_feature_ids,
    shogi_move_feature_ids,
)
from intrep.worlds.shogi.usi import shogi_usi_move_parts


class ShogiMoveEncodingTest(unittest.TestCase):
    def test_encodes_normal_move(self) -> None:
        feature_ids = shogi_move_feature_ids("7g7f", turn=shogi.BLACK)

        self.assertEqual(feature_ids.dtype, torch.long)
        self.assertEqual(tuple(feature_ids.shape), (SHOGI_MOVE_FIELD_COUNT,))
        self.assertNotEqual(int(feature_ids[0].item()), NO_FROM_SQUARE_ID)
        self.assertEqual(int(feature_ids[2].item()), 0)
        self.assertEqual(int(feature_ids[3].item()), NO_DROP_PIECE_ID)

    def test_encodes_promotion(self) -> None:
        feature_ids = shogi_move_feature_ids("2b3c+", turn=shogi.BLACK)

        self.assertEqual(int(feature_ids[2].item()), 1)

    def test_encodes_drop_move(self) -> None:
        feature_ids = shogi_move_feature_ids("P*5e", turn=shogi.BLACK)

        self.assertEqual(int(feature_ids[0].item()), NO_FROM_SQUARE_ID)
        self.assertGreater(int(feature_ids[3].item()), NO_DROP_PIECE_ID)

    def test_legal_move_feature_ids_are_padded_to_max_legal_move_count(self) -> None:
        feature_ids = shogi_legal_move_feature_ids(("7g7f", "3c3d"), turn=shogi.BLACK, max_legal_move_count=4)

        self.assertEqual(tuple(feature_ids.shape), (4, SHOGI_MOVE_FIELD_COUNT))
        self.assertTrue(torch.equal(feature_ids[2], torch.zeros(SHOGI_MOVE_FIELD_COUNT, dtype=torch.long)))

    def test_white_to_move_squares_are_side_to_move_relative(self) -> None:
        black_feature_ids = shogi_move_feature_ids("3c3d", turn=shogi.BLACK)
        white_feature_ids = shogi_move_feature_ids("3c3d", turn=shogi.WHITE)

        self.assertEqual(int(white_feature_ids[0].item()), 80 - int(black_feature_ids[0].item()))
        self.assertEqual(int(white_feature_ids[1].item()), 80 - int(black_feature_ids[1].item()))

    def test_usi_move_parts_match_python_shogi(self) -> None:
        for move_usi in ("7g7f", "2b3c+", "P*5e", "8h2b+", "2i3g"):
            parsed = shogi.Move.from_usi(move_usi)

            from_square, to_square, promotion, drop_piece_type = shogi_usi_move_parts(move_usi)

            self.assertEqual(from_square, None if parsed.from_square is None else int(parsed.from_square))
            self.assertEqual(to_square, int(parsed.to_square))
            self.assertEqual(promotion, bool(parsed.promotion))
            self.assertEqual(drop_piece_type, None if parsed.drop_piece_type is None else int(parsed.drop_piece_type))


if __name__ == "__main__":
    unittest.main()
